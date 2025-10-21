import express from "express"
import mysql from "mysql2/promise"
import fs from "fs"
import path from "path"
import dotenv from "dotenv"
import modelsRouter from "./routes/models.js"

dotenv.config()

const app = express()
const PORT = 3000

const MODELS_DIR = path.join(process.cwd(), "models")
if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, {recursive: true})

app.use(express.json())

async function connectWithRetry(delay = 3000) {
    let lastLog = 0
    while (true) {
        try {
            const db = await mysql.createConnection({
                host: process.env.MYSQL_HOST || "mysql",
                user: process.env.MYSQL_USER || "skynet",
                password: process.env.MYSQL_PASSWORD || "skynet",
                database: process.env.MYSQL_DATABASE || "skynet",
                multipleStatements: true
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

app.get("/", (req, res) => {
    res.sendFile(path.join(process.cwd(), "upload.html"))
})

app.listen(PORT, () =>
    console.log(`Server running on http://localhost:${PORT}`)
)
