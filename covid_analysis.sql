CREATE DATABASE covid_analysis;
use covid_analysis;
select * from covid19_cleaned;

-- Top 10 Countries by Total covid19 cases
select location, max(total_cases)as Total_cases from covid19_cleaned
group by location
order by Total_cases desc
limit 10;

-- Top 10Countries by Total covid19 Death
select location , max(total_deaths) as Total_deaths from covid19_cleaned 
group by location
order by total_deaths desc
limit 10;

-- which continent recorded highest cumullative covid19 cases
select continent ,sum(country_cases) as total_cases from (
select continent ,max(total_cases) as country_cases from covid19_cleaned
where continent is not null
group  by continent,location) as T
group by continent 
order by total_cases desc;

-- continent recorded highest total covid19 deaths
select continent,sum(country_deaths) as total_deaths from(
select continent,location,max(total_deaths) as country_deaths from covid19_cleaned
where continent is not null
group by continent,location)as T
group by continent
order by total_deaths desc;

-- top 10 countries by vaccination rate
select location, round(max(people_fully_vaccinated)/max(population)*100,2)as vaccination_rate
from covid19_cleaned
group by location
order by vaccination_rate desc
limit 10;

-- monthly covid19 cases Trend
select year, month ,month_name,sum(new_cases)as toal_cases from covid19_cleaned
group by year,month,month_name
order by year,month ;

-- monthly covid19 deaths Trend
select year,month,month_name,sum(new_deaths)as total_deaths from covid19_cleaned
group by year,month,month_name
order by year,month;

-- Rank Countries by Total COVID-19 Cases
select location,max(total_cases)as total_cases,rank()
over(order by max(total_cases)desc)as country_rank
from covid19_cleaned
group by location;

-- Classify Countries by Risk Level
select location,max(total_cases)as total_cases,
case
 when max(total_cases)>=10000000 then 'high risk'
 when max(total_cases)>=1000000 then 'medium risk'
 else 'low risk'
 end risk_leve
 from covid19_cleaned
 group by location
 order by total_cases desc;
 
 -- Month-over-Month Growth in New Cases (LAG) used to check previous column
-- How did COVID-19 cases grow compared to the previous month?
 
with monthly_cases as(
select year,month,month_name,sum(new_cases)as Total_cases from covid19_cleaned
group by year,month,month_name
)
select year,month,month_name,total_Cases,Lag(total_cases)over(order by year,month)as previous_month_cases,
total_cases - lag(total_Cases) over(order by year,month)as monthly_growth
from monthly_cases;

-- Top 3 Countries by Cases in Each Continent (DENSE_RANK)
-- Which are the top three countries with the highest COVID-19 cases within each continent?
WITH country_cases AS (
    SELECT
        continent,
        location,
        MAX(total_cases) AS total_cases
    FROM covid19_cleaned
    WHERE continent IS NOT NULL
    GROUP BY continent, location
)

SELECT *
FROM (
    SELECT
        continent,
        location,
        total_cases,
        DENSE_RANK() OVER (
            PARTITION BY continent
            ORDER BY total_cases DESC
        ) AS country_rank
    FROM country_cases
) ranked
WHERE country_rank <= 3;

