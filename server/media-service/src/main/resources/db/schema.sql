-- Postgre SQL init, warning: create "metawebthree" database before start this script.

-- CREATE DATABASE IF NOT EXISTS metawebthree;
-- \c metawebthree;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;

$$ LANGUAGE plpgsql;


DROP TABLE IF EXISTS "Artwork";
DROP TABLE IF EXISTS "People_Type";
DROP TABLE IF EXISTS "People";
DROP TABLE IF EXISTS "Artwork_Tag";
DROP TABLE IF EXISTS "Artwork_Category";

CREATE TABLE "Artwork_Category" (
    "id" SERIAL PRIMARY KEY,
    "name" VARCHAR NOT NULL,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_video_category_updated_at
BEFORE UPDATE ON "Artwork_Category"
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE "Artwork_Tag" (
    "id" SERIAL PRIMARY KEY,
    "tag" VARCHAR NOT NULL,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "People" (
    "id" SERIAL PRIMARY KEY,
    "name" VARCHAR NOT NULL,
    "types" SMALLINT[] DEFAULT '{}', -- List of Table People_Type IDs
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "People_Type" (
    "id" SERIAL PRIMARY KEY,
    "type" VARCHAR NOT NULL, -- Such as Director, Editor, Actor, Voicer
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "Artwork" (
    "id" SERIAL PRIMARY KEY,
    "series" VARCHAR(255) DEFAULT '', -- Batman Prequel
    "title" VARCHAR NOT NULL, -- Batman Begins
    "cover" VARCHAR(255) DEFAULT '', -- cover image URL
    "link" VARCHAR(255) DEFAULT '',
    "subtitle" VARCHAR(255) DEFAULT '', -- Behind the scenes
    "season" SMALLINT DEFAULT 1,
    "episode" SMALLINT DEFAULT 1,
    "category_id" INTEGER REFERENCES "Artwork_Category" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    "tags" INTEGER[] DEFAULT '{}', -- List of Table Tag IDs
    "year_tag" SMALLINT DEFAULT 0,
    "acts" INTEGER[] DEFAULT '{}', -- List of Table People_Type type=Actor IDs
    "director" INTEGER REFERENCES "People" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_Artwork_updated_at
BEFORE UPDATE ON "Artwork"
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE FUNCTION update_year_tag_column()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.created_at IS NOT NULL THEN
        NEW.year_tag = EXTRACT(YEAR FROM NEW.created_at);
    END IF;
    RETURN NEW;
END;

$$ LANGUAGE plpgsql;

CREATE TRIGGER update_video_tag_created_at
BEFORE INSERT ON "Artwork"
FOR EACH ROW
EXECUTE FUNCTION update_year_tag_column();

CREATE INDEX idx_Artwork_updated_at ON "Artwork" ("updated_at");

CREATE INDEX idx_video_tags ON "Artwork" USING GIN (tags);