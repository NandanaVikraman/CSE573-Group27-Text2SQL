import dynet as dy
import numpy as np


class BeamHypothesis:
    def __init__(self, tokens, log_prob, decoder_state, prev_context):
        self.tokens = tokens
        self.log_prob = log_prob
        self.decoder_state = decoder_state
        self.prev_context = prev_context

    def __lt__(self, other):
        return self.log_prob > other.log_prob

    def score(self, length_penalty=0.6):
        length = len(self.tokens) + 1
        penalty = ((5.0 + length) / 6.0) ** length_penalty
        return self.log_prob / penalty


class Seq2SQLModel:
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_dim=256):
        self.model = dy.Model()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.enc_dim = 2 * hidden_dim
        self.dec_dim = 2 * hidden_dim
        self.attention_dim = hidden_dim

        self.src_lookup = self.model.add_lookup_parameters((src_vocab_size, embed_dim))
        self.tgt_lookup = self.model.add_lookup_parameters((tgt_vocab_size, embed_dim))

        self.enc_fwd_lstm = dy.LSTMBuilder(1, embed_dim, hidden_dim, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(1, embed_dim, hidden_dim, self.model)
        self.dec_lstm = dy.LSTMBuilder(1, embed_dim + self.enc_dim, self.dec_dim, self.model)

        self.W_dec = self.model.add_parameters((self.attention_dim, self.dec_dim))
        self.W_enc = self.model.add_parameters((self.attention_dim, self.enc_dim))
        self.v_att = self.model.add_parameters((self.attention_dim,))

        self.W_out = self.model.add_parameters((tgt_vocab_size, self.dec_dim + self.enc_dim))
        self.b_out = self.model.add_parameters((tgt_vocab_size,))

        self.W_gate = self.model.add_parameters((1, self.dec_dim + self.enc_dim))
        self.b_gate = self.model.add_parameters((1,))

        self.W_init_h = self.model.add_parameters((self.dec_dim, self.enc_dim))
        self.b_init_h = self.model.add_parameters((self.dec_dim,))
        self.W_init_c = self.model.add_parameters((self.dec_dim, self.enc_dim))
        self.b_init_c = self.model.add_parameters((self.dec_dim,))

    def encode(self, input_ids):
        embeddings = [self.src_lookup[tok_id] for tok_id in input_ids]

        fwd_state = self.enc_fwd_lstm.initial_state()
        fwd_outputs = fwd_state.transduce(embeddings)

        bwd_state = self.enc_bwd_lstm.initial_state()
        bwd_outputs_rev = bwd_state.transduce(list(reversed(embeddings)))
        bwd_outputs = list(reversed(bwd_outputs_rev))

        encoder_outputs = [
            dy.concatenate([fwd_out, bwd_out])
            for fwd_out, bwd_out in zip(fwd_outputs, bwd_outputs)
        ]
        enc_matrix = dy.concatenate_cols(encoder_outputs)
        final_enc_state = dy.concatenate([fwd_outputs[-1], bwd_outputs[0]])

        return encoder_outputs, enc_matrix, final_enc_state

    def _precompute_enc_projections(self, encoder_outputs):
        return [self.W_enc * h_enc for h_enc in encoder_outputs]

    def _build_copy_scatter(self, src_ext_ids, num_oov):
        extended_vocab_size = self.tgt_vocab_size + num_oov
        T = len(src_ext_ids)
        copy_scatter_np = np.zeros((extended_vocab_size, T), dtype=np.float32)
        for src_pos, ext_id in enumerate(src_ext_ids):
            copy_scatter_np[ext_id, src_pos] = 1.0
        return dy.inputTensor(copy_scatter_np)

    def attend(self, dec_hidden, encoder_outputs, enc_matrix, enc_projections):
        dec_proj = self.W_dec * dec_hidden

        scores = []
        for enc_proj in enc_projections:
            combined = dy.tanh(dec_proj + enc_proj)
            score = dy.dot_product(self.v_att, combined)
            scores.append(score)

        scores_vec = dy.concatenate(scores)
        attention_weights = dy.softmax(scores_vec)
        context = enc_matrix * attention_weights

        return context, attention_weights

    def _init_decoder_state(self, final_enc_state):
        init_h = dy.tanh(self.W_init_h * final_enc_state + self.b_init_h)
        init_c = dy.tanh(self.W_init_c * final_enc_state + self.b_init_c)
        return self.dec_lstm.initial_state([init_c, init_h])

    def _compute_extended_distribution(
        self,
        gen_logits,
        attention_weights,
        src_ext_ids,
        num_oov,
        copy_scatter,
        combined,
    ):
        gate_logit = self.W_gate * combined + self.b_gate
        p_copy_scalar = dy.logistic(gate_logit)
        p_gen_scalar = 1.0 - p_copy_scalar

        p_vocab = dy.softmax(gen_logits)
        if num_oov > 0:
            p_vocab = dy.concatenate([p_vocab, dy.zeros(num_oov)])

        p_gen_contrib = p_vocab * p_gen_scalar
        p_copy_dist = copy_scatter * attention_weights
        p_copy_contrib = p_copy_dist * p_copy_scalar

        return p_gen_contrib + p_copy_contrib

    def compute_loss(self, input_ids, target_ids, src_ext_ids, tgt_ext_ids, num_oov):
        dy.renew_cg()

        encoder_outputs, enc_matrix, final_enc_state = self.encode(input_ids)
        enc_projections = self._precompute_enc_projections(encoder_outputs)
        copy_scatter = self._build_copy_scatter(src_ext_ids, num_oov)

        decoder_state = self._init_decoder_state(final_enc_state)

        prev_context = dy.zeros(self.enc_dim)
        losses = []

        for t in range(len(target_ids) - 1):
            prev_token = target_ids[t]
            next_token_ext = tgt_ext_ids[t + 1]

            prev_embed = self.tgt_lookup[prev_token]
            decoder_input = dy.concatenate([prev_embed, prev_context])

            decoder_state = decoder_state.add_input(decoder_input)
            dec_hidden = decoder_state.output()

            context, attention_weights = self.attend(
                dec_hidden, encoder_outputs, enc_matrix, enc_projections
            )
            prev_context = context

            combined = dy.concatenate([dec_hidden, context])
            gen_logits = self.W_out * combined + self.b_out

            p_extended = self._compute_extended_distribution(
                gen_logits,
                attention_weights,
                src_ext_ids,
                num_oov,
                copy_scatter,
                combined,
            )

            prob_next = dy.pick(p_extended, next_token_ext)
            losses.append(-dy.log(prob_next + 1e-10))

        return dy.esum(losses)

    def decode_greedy(self, input_ids, bos_id, eos_id, src_ext_ids, num_oov, max_len=100):
        dy.renew_cg()

        encoder_outputs, enc_matrix, final_enc_state = self.encode(input_ids)
        enc_projections = self._precompute_enc_projections(encoder_outputs)
        copy_scatter = self._build_copy_scatter(src_ext_ids, num_oov)

        decoder_state = self._init_decoder_state(final_enc_state)

        prev_context = dy.zeros(self.enc_dim)
        current_token = bos_id
        output_ids = []

        for _ in range(max_len):
            if current_token < self.tgt_vocab_size:
                curr_embed = self.tgt_lookup[current_token]
            else:
                curr_embed = self.tgt_lookup[1]

            decoder_input = dy.concatenate([curr_embed, prev_context])
            decoder_state = decoder_state.add_input(decoder_input)
            dec_hidden = decoder_state.output()

            context, attention_weights = self.attend(
                dec_hidden, encoder_outputs, enc_matrix, enc_projections
            )
            prev_context = context

            combined = dy.concatenate([dec_hidden, context])
            gen_logits = self.W_out * combined + self.b_out

            p_extended = self._compute_extended_distribution(
                gen_logits,
                attention_weights,
                src_ext_ids,
                num_oov,
                copy_scatter,
                combined,
            )

            next_token = int(np.argmax(p_extended.npvalue()))
            if next_token == eos_id:
                break

            output_ids.append(next_token)
            current_token = next_token

        return output_ids

    def decode_beam(
        self,
        input_ids,
        bos_id,
        eos_id,
        src_ext_ids,
        num_oov,
        beam_size=5,
        max_len=100,
        length_penalty=0.6,
    ):
        dy.renew_cg()

        encoder_outputs, enc_matrix, final_enc_state = self.encode(input_ids)
        enc_projections = self._precompute_enc_projections(encoder_outputs)
        copy_scatter = self._build_copy_scatter(src_ext_ids, num_oov)

        initial_decoder_state = self._init_decoder_state(final_enc_state)

        initial_hyp = BeamHypothesis(
            tokens=[],
            log_prob=0.0,
            decoder_state=initial_decoder_state,
            prev_context=dy.zeros(self.enc_dim),
        )

        active_hyps = [initial_hyp]
        completed_hyps = []

        for _ in range(max_len):
            if not active_hyps:
                break

            all_candidates = []

            for hyp in active_hyps:
                current_token = bos_id if not hyp.tokens else hyp.tokens[-1]

                if current_token < self.tgt_vocab_size:
                    curr_embed = self.tgt_lookup[current_token]
                else:
                    curr_embed = self.tgt_lookup[1]

                decoder_input = dy.concatenate([curr_embed, hyp.prev_context])
                new_decoder_state = hyp.decoder_state.add_input(decoder_input)
                dec_hidden = new_decoder_state.output()

                context, attention_weights = self.attend(
                    dec_hidden, encoder_outputs, enc_matrix, enc_projections
                )

                combined = dy.concatenate([dec_hidden, context])
                gen_logits = self.W_out * combined + self.b_out

                p_extended = self._compute_extended_distribution(
                    gen_logits,
                    attention_weights,
                    src_ext_ids,
                    num_oov,
                    copy_scatter,
                    combined,
                )

                probs = np.clip(p_extended.npvalue(), 1e-10, 1.0)
                log_probs = np.log(probs)
                top_k_indices = np.argsort(log_probs)[-beam_size * 2 :][::-1]

                for next_token in top_k_indices:
                    next_token = int(next_token)
                    new_hyp = BeamHypothesis(
                        tokens=hyp.tokens + [next_token],
                        log_prob=hyp.log_prob + log_probs[next_token],
                        decoder_state=new_decoder_state,
                        prev_context=context,
                    )
                    if next_token == eos_id:
                        completed_hyps.append(new_hyp)
                    else:
                        all_candidates.append(new_hyp)

            all_candidates.sort(key=lambda h: h.score(length_penalty), reverse=True)
            active_hyps = all_candidates[:beam_size]

            if len(completed_hyps) >= beam_size:
                completed_hyps.sort(key=lambda h: h.score(length_penalty), reverse=True)
                if active_hyps and completed_hyps[0].score(length_penalty) >= active_hyps[0].score(length_penalty):
                    break

        completed_hyps.extend(active_hyps)

        if not completed_hyps:
            return []

        completed_hyps.sort(key=lambda h: h.score(length_penalty), reverse=True)
        best_hyp = completed_hyps[0]
        return [t for t in best_hyp.tokens if t != eos_id]
