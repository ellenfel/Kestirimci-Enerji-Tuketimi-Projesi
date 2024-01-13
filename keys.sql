WITH LastVeriWithRowNumber AS (
    SELECT
        name,
        TO_TIMESTAMP(ts / 1000) AS formatted_ts,
        merged_column,
        ROW_NUMBER() OVER (PARTITION BY name, key ORDER BY ts DESC) AS row_num
    FROM last_veri
)
SELECT
    name,
    key,
    TO_CHAR(formatted_ts, 'DD/MM/YYYY HH24:MI:SS') AS formatted_date,
    merged_column
FROM LastVeriWithRowNumber
WHERE row_num = 1;
