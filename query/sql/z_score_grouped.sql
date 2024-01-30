WITH stats AS (
    SELECT
        building_type,
        neighbourhood,
        AVG(sale_price) OVER (PARTITION BY building_type, neighbourhood) AS mean_sale_price,
        STDDEV(sale_price) OVER (PARTITION BY building_type, neighbourhood) AS stddev_sale_price
    FROM sales
)
-- Calculate the z-score for each row within each group
SELECT
    building_type,
    neighbourhood,
    sale_price,
    (sale_price - mean_sale_price) / stddev_sale_price AS z_score
FROM
    sales,
    stats;