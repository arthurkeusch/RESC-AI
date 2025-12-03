CREATE TABLE IF NOT EXISTS models
(
    id_model INT AUTO_INCREMENT,
    name     VARCHAR(255) UNIQUE,
    params   TEXT,
    filename VARCHAR(255) UNIQUE,
    size     BIGINT,
    PRIMARY KEY (id_model)
);

CREATE TABLE IF NOT EXISTS datasets
(
    id_datatset       INT AUTO_INCREMENT,
    name              TEXT,
    description       TEXT,
    is_conversational BOOLEAN,
    PRIMARY KEY (id_datatset)
);

CREATE TABLE IF NOT EXISTS prompts
(
    id_prompt   INT AUTO_INCREMENT,
    prompt      TEXT,
    id_datatset INT NOT NULL,
    PRIMARY KEY (id_prompt),
    FOREIGN KEY (id_datatset) REFERENCES datasets (id_datatset)
);

CREATE TABLE IF NOT EXISTS devices
(
    id_devices INT AUTO_INCREMENT,
    name       TEXT,
    PRIMARY KEY (id_devices)
);

CREATE TABLE IF NOT EXISTS prompts_results
(
    id_result  INT AUTO_INCREMENT,
    response   TEXT,
    id_prompt  INT NOT NULL,
    id_model   INT NOT NULL,
    id_devices INT NOT NULL,
    PRIMARY KEY (id_result),
    FOREIGN KEY (id_prompt) REFERENCES prompts (id_prompt),
    FOREIGN KEY (id_model) REFERENCES models (id_model),
    FOREIGN KEY (id_devices) REFERENCES devices (id_devices)
);
