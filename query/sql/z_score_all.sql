WITH stats AS (
    SELECT
        AVG(sale_price) AS mean_sale_price,
        STDDEV(sale_price) AS stddev_sale_price
    FROM sales
)
SELECT
    building_type,
    neighbourhood,
    sale_price,
    (sale_price - mean_sale_price) / stddev_sale_price AS z_score
FROM
    sales,
    stats;
