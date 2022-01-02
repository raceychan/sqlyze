select *
from housing
where total_rooms <5
and median_income >5


-- name = house_sql
with base as (
     select total_rooms, median_income, population
     from housing
     where total_rooms > 15
     and median_income < 5 
)

select * from base
