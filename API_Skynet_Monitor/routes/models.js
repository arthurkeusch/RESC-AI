import {Router} from "express"
import multer from "multer"
import fs from "fs"
import path from "path"

export default function modelsRouter({db, MODELS_DIR}) {
    const router = Router()

    const storage = multer.diskStorage({
        destination: (req, file, cb) => cb(null, MODELS_DIR),
        filename: (req, file, cb) => cb(null, file.originalname.replace(/\s+/g, "_")),
    })
    const upload = multer({storage})

    router.get("/", async (req, res) => {
        try {
            const [rows] = await db.execute("SELECT * FROM models")
            res.json(rows)
        } catch (err) {
            res.status(500).json({error: "Failed to fetch models: " + err.message})
        }
    })

    router.get("/:id", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM models WHERE id_model = ?",
                [req.params.id]
            )
            if (rows.length === 0) return res.status(404).json({error: "Model not found"})
            res.json(rows[0])
        } catch (err) {
            res.status(500).json({error: "Failed to fetch model info: " + err.message})
        }
    })

    router.post("/upload", upload.single("file"), async (req, res) => {
        try {
            const {name, params} = req.body
            const file = req.file
            if (!file || !name || !params)
                return res.status(400).json({error: "Missing file, name, or params."})

            const filePath = path.join(MODELS_DIR, file.filename)
            const size = fs.statSync(filePath).size

            await db.execute(
                `INSERT INTO models (id_model, name, params, filename, size)
                 VALUES (NULL, ?, ?, ?, ?)`,
                [name, params, file.filename, size]
            )

            const [row] = await db.execute("SELECT id_model FROM models WHERE name = ?", [name])

            res.json({
                message: "Model uploaded successfully",
                id_model: row[0].id_model,
                name,
                params,
                filename: file.filename,
                size,
            })
        } catch (err) {
            res.status(500).json({error: "Upload failed: " + err.message})
        }
    })

    router.put("/:id", upload.single("file"), async (req, res) => {
        try {
            const id = req.params.id
            const [rows] = await db.execute("SELECT * FROM models WHERE id_model = ?", [id])
            if (rows.length === 0) return res.status(404).json({error: "Model not found"})
            const current = rows[0]

            const {name, params} = req.body
            let newFilename = current.filename
            let newSize = current.size

            if (req.file) {
                newFilename = req.file.filename
                const newPath = path.join(MODELS_DIR, newFilename)
                if (!fs.existsSync(newPath)) {
                    return res.status(500).json({error: "Uploaded file missing on disk"})
                }
                newSize = fs.statSync(newPath).size
            }

            const fields = []
            const values = []
            if (typeof name !== "undefined" && name !== current.name) {
                fields.push("name = ?")
                values.push(name)
            }
            if (typeof params !== "undefined" && params !== current.params) {
                fields.push("params = ?")
                values.push(params)
            }
            if (req.file) {
                fields.push("filename = ?")
                values.push(newFilename)
                fields.push("size = ?")
                values.push(newSize)
            }

            if (fields.length === 0) {
                const [updated] = await db.execute("SELECT * FROM models WHERE id_model = ?", [id])
                return res.json(updated[0])
            }

            values.push(id)
            await db.execute(`UPDATE models
                              SET ${fields.join(", ")}
                              WHERE id_model = ?`, values)

            if (req.file && current.filename !== newFilename) {
                const oldPath = path.join(MODELS_DIR, current.filename)
                if (fs.existsSync(oldPath)) {
                    try {
                        fs.unlinkSync(oldPath)
                    } catch {
                    }
                }
            }

            const [updated] = await db.execute("SELECT * FROM models WHERE id_model = ?", [id])
            res.json(updated[0])
        } catch (err) {
            res.status(500).json({error: "Update failed: " + err.message})
        }
    })

    router.delete("/:id", async (req, res) => {
        try {
            const id = req.params.id
            const [rows] = await db.execute("SELECT * FROM models WHERE id_model = ?", [id])
            if (rows.length === 0) return res.status(404).json({error: "Model not found"})
            const row = rows[0]

            const filePath = path.join(MODELS_DIR, row.filename)
            if (fs.existsSync(filePath)) {
                try {
                    fs.unlinkSync(filePath)
                } catch {
                }
            }

            await db.execute("DELETE FROM models WHERE id_model = ?", [id])

            res.json({message: "Model deleted", id_model: id})
        } catch (err) {
            res.status(500).json({error: "Delete failed: " + err.message})
        }
    })

    router.get("/download/:id", async (req, res) => {
        try {
            const [rows] = await db.execute("SELECT * FROM models WHERE id_model = ?", [
                req.params.id,
            ])
            if (rows.length === 0) return res.status(404).json({error: "Model not found"})

            const row = rows[0]
            const filePath = path.join(MODELS_DIR, row.filename)
            if (!fs.existsSync(filePath))
                return res.status(404).json({error: "File not found on disk"})

            res.download(filePath, row.filename)
        } catch (err) {
            res.status(500).json({error: "Download failed: " + err.message})
        }
    })

    return router
}
