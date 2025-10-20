import express from "express"
import multer from "multer"
import mysql from "mysql2/promise"
import fs from "fs"
import path from "path"

const app = express()
const PORT = 3000

const MODELS_DIR = path.join(process.cwd(), "models")
if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, { recursive: true })

app.use(express.json())

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, MODELS_DIR),
    filename: (req, file, cb) => cb(null, file.originalname.replace(/\s+/g, "_")),
})
const upload = multer({ storage })

async function connectWithRetry(delay = 3000) {
    let lastLog = 0
    while (true) {
        try {
            const db = await mysql.createConnection({
                host: process.env.MYSQL_HOST || "mysql",
                user: process.env.MYSQL_USER || "skynet",
                password: process.env.MYSQL_PASSWORD || "skynet",
                database: process.env.MYSQL_DATABASE || "skynet",
            })
            console.log("Connected to MySQL")
            return db
        } catch {
            const now = Date.now()
            if (now - lastLog > 30000) {
                console.log("Waiting for MySQL...")
                lastLog = now
            }
            await new Promise(res => setTimeout(res, delay))
        }
    }
}

const db = await connectWithRetry()

await db.execute(`
    CREATE TABLE IF NOT EXISTS models (
                                          id INT AUTO_INCREMENT PRIMARY KEY,
                                          name VARCHAR(255) UNIQUE,
                                          params TEXT,
                                          filename VARCHAR(255),
                                          size BIGINT,
                                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
`)

app.post("/upload", upload.single("file"), async (req, res) => {
    try {
        const { name, params } = req.body
        const file = req.file
        if (!file || !name || !params)
            return res.status(400).json({ error: "Missing file, name, or params." })
        const filePath = path.join(MODELS_DIR, file.filename)
        const size = fs.statSync(filePath).size
        await db.execute(
            `INSERT INTO models (name, params, filename, size)
             VALUES (?, ?, ?, ?)
             ON DUPLICATE KEY UPDATE params=VALUES(params), filename=VALUES(filename), size=VALUES(size)`,
            [name, params, file.filename, size]
        )
        res.json({ message: "Model uploaded successfully", name, params, size })
    } catch {
        res.status(500).json({ error: "Upload failed" })
    }
})

app.get("/models", async (req, res) => {
    try {
        const [rows] = await db.execute(
            "SELECT name, params, size, filename, created_at FROM models ORDER BY created_at DESC"
        )
        res.json(rows)
    } catch {
        res.status(500).json({ error: "Failed to fetch models" })
    }
})

app.get("/download/:name", async (req, res) => {
    try {
        const [rows] = await db.execute("SELECT * FROM models WHERE name = ?", [
            req.params.name,
        ])
        if (rows.length === 0)
            return res.status(404).json({ error: "Model not found" })
        const row = rows[0]
        const filePath = path.join(MODELS_DIR, row.filename)
        if (!fs.existsSync(filePath))
            return res.status(404).json({ error: "File not found on disk" })
        res.download(filePath, row.filename)
    } catch {
        res.status(500).json({ error: "Download failed" })
    }
})

app.get("/model/:name", async (req, res) => {
    try {
        const [rows] = await db.execute(
            "SELECT name, params, size, filename, created_at FROM models WHERE name = ?",
            [req.params.name]
        )
        if (rows.length === 0)
            return res.status(404).json({ error: "Model not found" })
        res.json(rows[0])
    } catch {
        res.status(500).json({ error: "Failed to fetch model info" })
    }
})

app.get("/", (req, res) => {
    res.sendFile(path.join(process.cwd(), "upload.html"))
})

app.listen(PORT, () =>
    console.log(`Server running on http://localhost:${PORT}`)
)
