CREATE TABLE IF NOT EXISTS models
(
    id       INT AUTO_INCREMENT PRIMARY KEY,
    name     VARCHAR(255) UNIQUE,
    params   TEXT,
    filename VARCHAR(255) UNIQUE,
    size     BIGINT
)