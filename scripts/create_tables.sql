CREATE TABLE IF NOT EXISTS tags (
    id BIGINT PRIMARY KEY,
    tag_name TEXT NOT NULL,
    count INTEGER
);

CREATE TABLE IF NOT EXISTS posts (
    id BIGINT PRIMARY KEY,
    post_type_id INTEGER NOT NULL,
    parent_id BIGINT,
    accepted_answer_id BIGINT,
    creation_date TIMESTAMP,
    score INTEGER,
    view_count INTEGER,
    answer_count INTEGER,
    comment_count INTEGER,
    favorite_count INTEGER,
    title TEXT,
    body TEXT,
    tags TEXT
);

CREATE TABLE IF NOT EXISTS votes (
    id BIGINT PRIMARY KEY,
    post_id BIGINT NOT NULL,
    vote_type_id INTEGER NOT NULL,
    creation_date DATE,
    bounty_amount INTEGER
);