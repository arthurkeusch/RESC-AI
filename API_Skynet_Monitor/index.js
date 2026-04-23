import express from "express"
import mysql from "mysql2/promise"
import fs from "fs"
import path from "path"
import dotenv from "dotenv"
import modelsRouter from "./routes/models.js"
import promptsRouter from "./routes/prompts.js"
import datasetsRouter from "./routes/datasets.js"
import heimdallRouter from "./routes/heimdall.js"
import devicesRouter from "./routes/devices.js"
import statsRouter from "./routes/stats.js"

dotenv.config()

const app = express()
const PORT = 3000

const MODELS_DIR = path.join(process.cwd(), "models")
if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, {recursive: true})

app.use(express.json({limit: "10mb"}))

function createDbPool() {
    return mysql.createPool({
        host: process.env.MYSQL_HOST || "mysql",
        user: process.env.MYSQL_USER || "skynet",
        password: process.env.MYSQL_PASSWORD || "skynet",
        database: process.env.MYSQL_DATABASE || "skynet",
        multipleStatements: true,
        waitForConnections: true,
        connectionLimit: Number(process.env.MYSQL_CONNECTION_LIMIT || 10),
        queueLimit: 0,
        enableKeepAlive: true,
        keepAliveInitialDelay: 0,
    })
}

async function waitForDatabase(db, delay = 3000) {
    let lastLog = 0
    while (true) {
        try {
            await db.query("SELECT 1")
            console.log("Connected to MySQL")
            return
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

const db = createDbPool()
await waitForDatabase(db)

const sqlPath = path.join(process.cwd(), "bdd.sql")
if (fs.existsSync(sqlPath)) {
    const sqlContent = fs.readFileSync(sqlPath, "utf8")
    try {
        await db.query(sqlContent)
        console.log("bdd.sql executed successfully")
    } catch (err) {
        console.error("Error executing bdd.sql:", err)
        process.exit(1)
    }
} else {
    console.error("bdd.sql not found")
    process.exit(1)
}

app.use("/models", modelsRouter({db, MODELS_DIR}))
app.use("/datasets", datasetsRouter({db}))
app.use("/prompts", promptsRouter({db}))
app.use("/heimdall", heimdallRouter({db}))
app.use("/devices", devicesRouter({db}))
app.use("/stats", statsRouter({db}))

app.get("/health", async (req, res) => {
    try {
        await db.query("SELECT 1")
        res.json({status: "ok", database: "ok"})
    } catch (err) {
        console.error("Healthcheck failed:", err)
        res.status(503).json({status: "error", database: "unavailable"})
    }
})

app.get("/", (req, res) => {
    res.sendFile(path.join(process.cwd(), "upload.html"))
})

const server = app.listen(PORT, () =>
    console.log(`Server running on http://localhost:${PORT}`)
)

async function shutdown(signal) {
    console.log(`${signal} received, shutting down...`)
    server.close(async () => {
        try {
            await db.end()
        } catch (err) {
            console.error("Error while closing MySQL pool:", err)
        } finally {
            process.exit(0)
        }
    })
}

process.on("SIGTERM", () => shutdown("SIGTERM"))
process.on("SIGINT", () => shutdown("SIGINT"))
