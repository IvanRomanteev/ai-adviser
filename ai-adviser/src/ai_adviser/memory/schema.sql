DELETE FROM messages;
DELETE FROM summaries;
DELETE FROM sqlite_sequence WHERE name IN ('messages','summaries');
-- (опционально) чтобы физически уменьшить файл базы:
VACUUM;
