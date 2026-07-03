import { useState, useRef } from "react";

function highlightSQL(sql) {
  if (!sql) return "";
  const keywords = /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|INDEX|DROP|ALTER|ADD|COLUMN|PRIMARY|KEY|FOREIGN|REFERENCES|DISTINCT|AS|AND|OR|NOT|IN|IS|NULL|LIKE|BETWEEN|EXISTS|UNION|ALL|CASE|WHEN|THEN|ELSE|END|WITH|BY|ASC|DESC|COUNT|SUM|AVG|MIN|MAX|COALESCE|CAST|CONVERT)\b/gi;
  const strings = /('[^']*'|"[^"]*")/g;
  const numbers = /\b(\d+(\.\d+)?)\b/g;
  const comments = /(--[^\n]*|\/\*[\s\S]*?\*\/)/g;

  return sql
    .replace(comments, '<span class="sql-comment">$1</span>')
    .replace(strings, '<span class="sql-string">$1</span>')
    .replace(keywords, '<span class="sql-keyword">$1</span>')
    .replace(numbers, '<span class="sql-number">$1</span>');
}

const SAMPLE_SCHEMA = `CREATE TABLE customers (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  created_at TIMESTAMP
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customers(id),
  total DECIMAL(10,2),
  status VARCHAR(20),
  created_at TIMESTAMP
);

CREATE TABLE products (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  price DECIMAL(10,2),
  stock INT
);`;

const SAMPLE_QUERIES = [
  "Show me total sales per customer this month",
  "Find all orders with status pending",
  "Which products have stock below 10?",
  "Top 5 customers by order value",
];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export default function Text2SQLUI() {
  const [nlQuery, setNlQuery] = useState("");
  const [schema, setSchema] = useState(SAMPLE_SCHEMA);
  const [generatedSQL, setGeneratedSQL] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [history, setHistory] = useState([]);
  const [copied, setCopied] = useState(false);
  const [schemaInput, setSchemaInput] = useState("paste");
  const [useModulo, setUseModulo] = useState(false);

  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleGenerate = async () => {
    if (!nlQuery.trim()) return;
    if (!schema.trim()) {
      setErrorMessage("Please provide schema DDL before generating SQL.");
      return;
    }

    setIsLoading(true);
    setGeneratedSQL("");
    setErrorMessage("");

    try {
      const response = await fetch(`${API_BASE_URL}/generate-sql`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: nlQuery,
          schema_ddl: schema,
          use_modulo: useModulo,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to generate SQL.");
      }

      const sql = payload.sql ?? "";
      setGeneratedSQL(sql);
      setHistory((prev) => [
        {
          query: nlQuery,
          sql,
          mode: payload.used_modulo ? "Modulo" : "Single",
          time: new Date().toLocaleTimeString(),
        },
        ...prev.slice(0, 9),
      ]);

      if (!sql) {
        setErrorMessage("The model returned an empty SQL result. Try rephrasing your question.");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Request failed.";
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedSQL);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSampleClick = (query) => setNlQuery(query);

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleGenerate();
  };

  const handleSchemaFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileText = await file.text();
    setSchema(fileText);
    setSchemaInput("paste");
    e.target.value = "";
  };

  return (
    <div style={styles.root}>
      <style>{css}</style>

      <aside style={styles.sidebar}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>#</span>
          <span style={styles.logoText}>
            text<span style={styles.logoAccent}>2sql</span>
          </span>
        </div>

        <nav style={styles.nav}>
          <button style={{ ...styles.navItem, ...styles.navItemActive }}>
            <span>Q</span>
            Query
          </button>
        </nav>
      </aside>

      <main style={styles.main}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.headerTitle}>Query Builder</h1>
            <p style={styles.headerSub}>Translate natural language into SQL instantly</p>
          </div>
        </header>

        <div style={styles.modeRow}>
          <label style={styles.toggleLabel}>
            <input
              type="checkbox"
              checked={useModulo}
              onChange={(e) => setUseModulo(e.target.checked)}
            />
            Use LLM-Modulo verifier loop
          </label>
          <span style={styles.modeHint}>{useModulo ? "Higher quality, slower" : "Fast single-pass mode"}</span>
        </div>

        <div style={styles.grid}>
          <div style={styles.col}>
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <span style={styles.cardLabel}>Natural Language Query</span>
                <span style={styles.hint}>Ctrl/Cmd + Enter to generate</span>
              </div>
              <textarea
                ref={textareaRef}
                value={nlQuery}
                onChange={(e) => setNlQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about your data..."
                style={styles.nlTextarea}
                rows={4}
              />
              <div style={styles.samples}>
                {SAMPLE_QUERIES.map((q) => (
                  <button key={q} onClick={() => handleSampleClick(q)} style={styles.sampleChip}>
                    {q}
                  </button>
                ))}
              </div>
            </div>

            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <span style={styles.cardLabel}>Database Schema</span>
                <div style={styles.tabRow}>
                  {["paste", "upload"].map((t) => (
                    <button
                      key={t}
                      onClick={() => setSchemaInput(t)}
                      style={{
                        ...styles.tabBtn,
                        ...(schemaInput === t ? styles.tabBtnActive : {}),
                      }}
                    >
                      {t === "paste" ? "Paste DDL" : "Upload .sql"}
                    </button>
                  ))}
                </div>
              </div>

              {schemaInput === "paste" ? (
                <textarea
                  value={schema}
                  onChange={(e) => setSchema(e.target.value)}
                  style={styles.schemaTextarea}
                  rows={12}
                  spellCheck={false}
                />
              ) : (
                <div style={styles.uploadZone} onClick={() => fileInputRef.current?.click()}>
                  <span style={styles.uploadIcon}>^</span>
                  <p style={styles.uploadText}>Drop your .sql file here</p>
                  <p style={styles.uploadSub}>or click to browse</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".sql,text/plain"
                    style={styles.uploadInput}
                    onChange={handleSchemaFileUpload}
                  />
                </div>
              )}
            </div>
          </div>

          <div style={styles.col}>
            <button
              onClick={handleGenerate}
              disabled={isLoading || !nlQuery.trim() || !schema.trim()}
              style={{
                ...styles.generateBtn,
                ...(isLoading || !nlQuery.trim() || !schema.trim() ? styles.generateBtnDisabled : {}),
              }}
              className="generate-btn"
            >
              {isLoading ? <span style={styles.spinner} className="spinner" /> : "Generate SQL ->"}
            </button>

            {errorMessage && <div style={styles.errorBox}>{errorMessage}</div>}

            <div style={{ ...styles.card, flex: 1 }}>
              <div style={styles.cardHeader}>
                <span style={styles.cardLabel}>Generated SQL</span>
                {generatedSQL && (
                  <button onClick={handleCopy} style={styles.copyBtn}>
                    {copied ? "Copied" : "Copy"}
                  </button>
                )}
              </div>

              {isLoading ? (
                <div style={styles.loadingBlock}>
                  <div style={styles.loadingBar} className="loading-bar" />
                  <div style={{ ...styles.loadingBar, width: "70%", animationDelay: "0.15s" }} className="loading-bar" />
                  <div style={{ ...styles.loadingBar, width: "85%", animationDelay: "0.3s" }} className="loading-bar" />
                </div>
              ) : generatedSQL ? (
                <pre style={styles.sqlPre} dangerouslySetInnerHTML={{ __html: highlightSQL(generatedSQL) }} />
              ) : (
                <div style={styles.emptyState}>
                  <span style={styles.emptyIcon}>#</span>
                  <p style={styles.emptyText}>Your SQL will appear here</p>
                </div>
              )}
            </div>

            {history.length > 0 && (
              <div style={styles.card}>
                <div style={styles.cardHeader}>
                  <span style={styles.cardLabel}>Session History</span>
                </div>
                <div style={styles.historyList}>
                  {history.slice(0, 5).map((item, i) => (
                    <button
                      key={i}
                      style={styles.historyItem}
                      onClick={() => {
                        setNlQuery(item.query);
                        setGeneratedSQL(item.sql);
                      }}
                    >
                      <span style={styles.historyQuery}>{item.query}</span>
                      <span style={styles.historyMeta}>{item.mode} | {item.time}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>
      </main>
    </div>
  );
}

const ACCENT = "#00E5A0";
const ACCENT2 = "#0057FF";
const BG = "#0B0F1A";
const SURFACE = "#131929";
const SURFACE2 = "#1A2235";
const BORDER = "#1E2D45";
const TEXT = "#E2EAF4";
const MUTED = "#4A6080";
const FONT_MONO = "'JetBrains Mono', 'Fira Code', monospace";
const FONT_DISPLAY = "'DM Sans', 'Outfit', sans-serif";

const styles = {
  root: {
    display: "flex",
    minHeight: "100vh",
    background: BG,
    color: TEXT,
    fontFamily: FONT_DISPLAY,
  },
  sidebar: {
    width: 220,
    background: SURFACE,
    borderRight: `1px solid ${BORDER}`,
    display: "flex",
    flexDirection: "column",
    padding: "24px 0",
    flexShrink: 0,
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "0 24px 32px",
    borderBottom: `1px solid ${BORDER}`,
    marginBottom: 16,
  },
  logoIcon: { fontSize: 22, color: ACCENT },
  logoText: { fontSize: 18, fontWeight: 700, letterSpacing: "-0.5px" },
  logoAccent: { color: ACCENT },
  nav: { display: "flex", flexDirection: "column", gap: 4, padding: "0 12px" },
  navItem: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "10px 14px",
    borderRadius: 8,
    border: "none",
    background: "transparent",
    color: MUTED,
    fontSize: 14,
    fontFamily: FONT_DISPLAY,
    cursor: "pointer",
    textAlign: "left",
    transition: "all 0.15s",
  },
  navItemActive: {
    background: `${ACCENT}15`,
    color: ACCENT,
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    padding: 32,
    overflow: "auto",
    gap: 18,
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerTitle: {
    fontSize: 26,
    fontWeight: 700,
    margin: 0,
    letterSpacing: "-0.5px",
  },
  headerSub: {
    fontSize: 13,
    color: MUTED,
    margin: "4px 0 0",
  },
  modeRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    background: SURFACE,
    border: `1px solid ${BORDER}`,
    borderRadius: 10,
    padding: "10px 14px",
  },
  toggleLabel: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 13,
    color: TEXT,
    cursor: "pointer",
  },
  modeHint: {
    fontSize: 12,
    color: MUTED,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 20,
    flex: 1,
  },
  col: {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    minWidth: 0,
  },
  card: {
    background: SURFACE,
    border: `1px solid ${BORDER}`,
    borderRadius: 12,
    padding: 20,
    display: "flex",
    flexDirection: "column",
    gap: 14,
    minWidth: 0,
  },
  cardHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  cardLabel: {
    fontSize: 11,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.1em",
    color: MUTED,
  },
  hint: { fontSize: 11, color: MUTED },
  nlTextarea: {
    background: SURFACE2,
    border: `1px solid ${BORDER}`,
    borderRadius: 8,
    color: TEXT,
    fontFamily: FONT_DISPLAY,
    fontSize: 15,
    padding: "14px 16px",
    resize: "vertical",
    outline: "none",
    lineHeight: 1.6,
    transition: "border-color 0.2s",
  },
  samples: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
  },
  sampleChip: {
    background: SURFACE2,
    border: `1px solid ${BORDER}`,
    color: MUTED,
    borderRadius: 20,
    padding: "5px 12px",
    fontSize: 12,
    cursor: "pointer",
    fontFamily: FONT_DISPLAY,
    transition: "all 0.15s",
  },
  tabRow: { display: "flex", gap: 4 },
  tabBtn: {
    background: "transparent",
    border: `1px solid ${BORDER}`,
    color: MUTED,
    padding: "4px 10px",
    borderRadius: 6,
    fontSize: 11,
    cursor: "pointer",
    fontFamily: FONT_DISPLAY,
    transition: "all 0.15s",
  },
  tabBtnActive: {
    background: `${ACCENT}15`,
    borderColor: `${ACCENT}50`,
    color: ACCENT,
  },
  schemaTextarea: {
    background: "#0D1520",
    border: `1px solid ${BORDER}`,
    borderRadius: 8,
    color: "#7EC8A4",
    fontFamily: FONT_MONO,
    fontSize: 12,
    padding: "14px 16px",
    resize: "vertical",
    outline: "none",
    lineHeight: 1.7,
  },
  uploadZone: {
    border: `2px dashed ${BORDER}`,
    borderRadius: 8,
    padding: "36px 20px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
    cursor: "pointer",
  },
  uploadIcon: { fontSize: 28, color: MUTED },
  uploadText: { fontSize: 14, color: TEXT, margin: 0 },
  uploadSub: { fontSize: 12, color: MUTED, margin: 0 },
  uploadInput: { display: "none" },
  generateBtn: {
    background: `linear-gradient(135deg, ${ACCENT}, ${ACCENT2})`,
    border: "none",
    color: "#000",
    fontFamily: FONT_DISPLAY,
    fontWeight: 700,
    fontSize: 15,
    padding: "14px",
    borderRadius: 10,
    cursor: "pointer",
    letterSpacing: "0.02em",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    transition: "opacity 0.2s, transform 0.1s",
  },
  generateBtnDisabled: {
    opacity: 0.4,
    cursor: "not-allowed",
  },
  spinner: {
    width: 18,
    height: 18,
    border: "2px solid #00000040",
    borderTop: "2px solid #000",
    borderRadius: "50%",
    display: "inline-block",
  },
  errorBox: {
    background: "#3D1D28",
    border: "1px solid #7A3346",
    color: "#FFB3C3",
    padding: "10px 12px",
    borderRadius: 8,
    fontSize: 13,
  },
  copyBtn: {
    background: `${ACCENT}15`,
    border: `1px solid ${ACCENT}40`,
    color: ACCENT,
    padding: "4px 12px",
    borderRadius: 6,
    fontSize: 12,
    cursor: "pointer",
    fontFamily: FONT_DISPLAY,
  },
  sqlPre: {
    background: "#0D1520",
    border: `1px solid ${BORDER}`,
    borderRadius: 8,
    padding: "16px",
    fontFamily: FONT_MONO,
    fontSize: 13,
    lineHeight: 1.75,
    overflow: "auto",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    overflowWrap: "anywhere",
    margin: 0,
    flex: 1,
    minWidth: 0,
  },
  loadingBlock: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    padding: "16px",
  },
  loadingBar: {
    height: 14,
    background: SURFACE2,
    borderRadius: 6,
    width: "100%",
  },
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "48px 20px",
    gap: 12,
    opacity: 0.3,
  },
  emptyIcon: { fontSize: 36 },
  emptyText: { fontSize: 14, margin: 0, color: MUTED },
  historyList: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  historyItem: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
    background: SURFACE2,
    border: `1px solid ${BORDER}`,
    borderRadius: 8,
    padding: "10px 14px",
    cursor: "pointer",
    textAlign: "left",
    transition: "border-color 0.15s",
    fontFamily: FONT_DISPLAY,
  },
  historyQuery: {
    fontSize: 13,
    color: TEXT,
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  historyMeta: {
    fontSize: 11,
    color: MUTED,
  },
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  * { box-sizing: border-box; }
  body { margin: 0; }

  .sql-keyword { color: #7AB4F5; font-weight: 600; }
  .sql-string  { color: #F1A266; }
  .sql-number  { color: #B5CEA8; }
  .sql-comment { color: #5A7A5A; font-style: italic; }

  textarea:focus {
    border-color: #00E5A040 !important;
    box-shadow: 0 0 0 3px #00E5A010;
  }

  .generate-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px #00E5A030;
  }

  .spinner {
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .loading-bar {
    animation: pulse 1.2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.7; }
  }

  button:hover .sql-keyword { opacity: 0.9; }
`;
