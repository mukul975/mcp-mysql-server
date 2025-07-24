# 🚀 MySQL MCP Server — AI-Driven MySQL Control & Diagnostics for LLMs

A fully featured **Model Context Protocol (MCP) server for MySQL** — designed to empower **AI assistants, LLMs (like ChatGPT, Claude, Gemini)**, and automation tools with deep **MySQL database access**, diagnostics, and intelligent control.

> ⚡ Ideal for building **AI-powered database agents**, DevOps automation, or managing **MySQL with natural language**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![MySQL Support](https://img.shields.io/badge/MySQL-5.7%20%7C%208.x-blue.svg)](https://www.mysql.com/)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)](https://github.com/mukul975/mcp-mysql-server)

---

## 📆 Repository Info

* **GitHub**: [mcp-mysql-server](https://github.com/mukul975/mcp-mysql-server)
* **Author**: [@mukul975](https://github.com/mukul975)
* **License**: MIT
* **Keywords**: `MySQL`, `MCP Server`, `AI MySQL interface`, `LLM database tool`, `MySQL automation`, `chatbot SQL`

---

## 🧠 Key Features

This MySQL MCP Server provides **LLMs and AI tools** with:

* ✅ **Secure Query Execution**: Handles `SELECT`, `INSERT`, `UPDATE`, `DELETE`, etc. with read-only validation.
* 🔍 **Schema Introspection**: List tables, inspect structures, indexes, keys.
* 📊 **Performance Diagnostics**: Analyze fragmentation, slow queries, buffer pool stats.
* 🛡️ **Security Auditing**: Manage users, audit SSL, monitor roles & privileges.
* 🧩 **Backup & Replication Monitor**: View replication lag, log status, recovery readiness.
* ⚙️ **System Monitoring**: Get process list, memory usage, plugin state.
* 📈 **Query Insights**: Index recommendations, event scheduler overview.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
# Or install manually:
pip install "mcp[cli]"
pip install mysql-connector-python
```

---

## ⚙️ Environment Configuration

| Variable         | Description           | Default     |
| ---------------- | --------------------- | ----------- |
| `MYSQL_HOST`     | MySQL server hostname | `localhost` |
| `MYSQL_PORT`     | MySQL port            | `3306`      |
| `MYSQL_USER`     | Username              | `root`      |
| `MYSQL_PASSWORD` | Password              | (empty)     |
| `MYSQL_DATABASE` | Target DB             | (empty)     |

> 💡 Use `.env` or export variables manually

---

## ▶️ Run & Usage

```bash
python mysql_server.py                # Default (stdin transport)
python mysql_server.py --transport sse  # For web clients (SSE)
python mysql_server.py --help        # Command help
```

---

## 🔗 Claude Desktop Integration

```json
{
  "mcpServers": {
    "mysql": {
      "command": "python",
      "args": ["path/to/mysql_server.py"],
      "env": {
        "MYSQL_HOST": "localhost",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "your_user",
        "MYSQL_PASSWORD": "your_pass",
        "MYSQL_DATABASE": "your_db"
      }
    }
  }
}
```

---

## 🧪 Example Workflows

* `list_tables`: Lists all tables
* `describe_table`: Schema of a specific table
* `execute_sql`: Run select or data modification queries
* `mysql_slow_query_analysis`: Detect slow queries
* `mysql_user_privileges`: Audit user access
* `mysql_replication_lag_monitoring`: Check lag in replication

---

## 🧰 Toolset Highlights

> Access tools via code or LLM prompts:

* `mysql_query`, `list_mysql_tables`, `mysql_table_schema`
* `mysql_index_optimization_suggestions`, `mysql_deadlock_detection`
* `mysql_ssl_tls_configuration_audit`, `mysql_backup_health_check`
* `mysql_server_health_dashboard`, `mysql_event_scheduler`, and **dozens more**

---

## 🔐 Security Considerations

* Store secrets in env vars or vault
* Do not expose server publicly
* Privilege validation and SSL audit included

---

## 🧪 Error Handling

* Connection errors
* SQL syntax issues
* Network timeouts
* Returns structured error response

---

## 💡 Development Structure

```text
mcp-mysql-server/
├── mysql_server.py        # Entry point
├── requirements.txt       # Dependencies
├── README.md              # Docs
└── pyproject.toml         # Project metadata (optional)
```

---

## ✅ Testing Steps

1. Ensure MySQL is running
2. Set environment vars
3. Run `python mysql_server.py`
4. Try with `mcp dev mysql_server.py` (if using MCP CLI)

---

## 🤝 Contributing

* Fork the repo
* Create feature/bug branches
* Submit PRs with description & tests

---

## 📄 FAQ

**Q: What is MCP?**
A: Model Context Protocol (MCP) is an interface to give LLMs access to structured tools like databases, APIs, and system utilities.

**Q: Can I use this with ChatGPT or Claude?**
Yes! It's designed for direct integration with AI/LLM tools that support tool-use or system-level automation.

**Q: Is it safe to run this in production?**
It depends on your environment. Always restrict access, use read-only roles, and monitor logs.

---

## 📄 License

MIT License — Open source and free to use.

---

## 🔎 GitHub SEO Tips (apply on GitHub)


---

## 📣 Promote It

* Share on [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
* Post to LinkedIn or Twitter with GIF or screenshots
* Submit to [awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) or similar curated lists

---

Let me know if you need an HTML version, web preview, or GitHub Pages site for this!
