import { Router } from "express"
import multer from "multer"
import fs from "fs"
import path from "path"

export default function modelsRouter({ db, MODELS_DIR }) {
    const router = Router()

    const storage = multer.diskStorage({
        destination: (req, file, cb) => cb(null, MODELS_DIR),
        filename: (req, file, cb) => cb(null, file.originalname.replace(/\s+/g, "_")),
    })
    const upload = multer({ storage })

    router.get("/", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM models"
            )
            res.json(rows)
        } catch (err) {
            res.status(500).json({ error: "Failed to fetch models" })
        }
    })

    router.get("/:id", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM models WHERE id = ?",
                [req.params.id]
            )
            if (rows.length === 0) return res.status(404).json({ error: "Model not found" })
            res.json(rows[0])
        } catch (err) {
            res.status(500).json({ error: "Failed to fetch model info" })
        }
    })

    router.post("/upload", upload.single("file"), async (req, res) => {
        try {
            const { name, params } = req.body
            const file = req.file
            if (!file || !name || !params)
                return res.status(400).json({ error: "Missing file, name, or params." })

            const filePath = path.join(MODELS_DIR, file.filename)
            const size = fs.statSync(filePath).size

            await db.execute(
                `INSERT INTO models (name, params, filename, size)
                 VALUES (?, ?, ?, ?)`,
                [name, params, file.filename, size]
            )

            const [row] = await db.execute("SELECT id FROM models WHERE name = ?", [name])

            res.json({
                message: "Model uploaded successfully",
                id: row[0].id,
                name,
                params,
                size,
            })
        } catch (err) {
            res.status(500).json({ error: "Upload failed" })
        }
    })

    router.get("/download/:id", async (req, res) => {
        try {
            const [rows] = await db.execute("SELECT * FROM models WHERE id = ?", [
                req.params.id,
            ])
            if (rows.length === 0) return res.status(404).json({ error: "Model not found" })

            const row = rows[0]
            const filePath = path.join(MODELS_DIR, row.filename)
            if (!fs.existsSync(filePath))
                return res.status(404).json({ error: "File not found on disk" })

            res.download(filePath, row.filename)
        } catch (err) {
            res.status(500).json({ error: "Download failed" })
        }
    })

    return router
}
