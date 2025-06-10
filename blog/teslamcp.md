# How I Connected My Tesla to Claude

_Ever wondered what happens when you combine Tesla's API, PostgreSQL, and Claude? You get a conversational car assistant that can tell you everything from battery levels to efficiency patterns. Here's how I built it._

![teslamcp.gif](https://github.com/cobanov/blog-renderer/blob/main/assets/teslamcp.gif?raw=true)

In the digital world, I'm obsessed with two things: first, making my electronic devices talk to each other (even when they don't need to communicate, or when it's downright absurd), and second, having them log everything they do. When my doorbell rings, what hours my TV is on, how much data each device uses, what temperature my fridge is at right now, how many notifications I get from each app throughout the day – I'm curious about everything. Somehow, having statistics on everything, knowing that data is being collected even when I don't need it, watching it all flow around feels like watching a beautiful aquarium to me.

![little_cobanov.png](https://github.com/cobanov/blog-renderer/blob/main/assets/little_cobanov.png?raw=true)

I'm not kidding when I say that one of the biggest reasons I bought a Tesla was because the car had API access. After all, every car gets you from point A to point B, but the idea of buying something I could tinker with had me hooked. Since I was 9 years old, I've been taking apart everything, modifying software, and trying to do as much tinkering as possible beyond just using things as intended – until I break them. Tesla allowing this was fantastic. Even before I bought the car, I immediately jumped on GitHub to research what I could do with it. When I discovered the [TeslaMate](https://github.com/teslamate-org/teslamate) application, I knew this was exactly what I was looking for – I'd use it as a data collector and storage system for this project!

Simply put, TeslaMate logs all of your car's data to PostgreSQL using the car's API access, with all necessary visualizations handled through Grafana. Since I've been familiar with both technologies for years, I got it up and running pretty easily with a simple `docker-compose` file.

I have a 4-bay NAS server at home that runs 24/7 and handles all my self-hosting needs, so everything came together very quickly.

![teslamate_dashboard.png](https://github.com/cobanov/blog-renderer/blob/main/assets/teslamate_dashboard.png?raw=true)
_The Grafana dashboards that TeslaMate provides out of the box_

This logging setup was amazing at first, but after a while, it didn't mean much to me beyond providing a nice dashboard. The excitement of logging into Grafana to check things wore off pretty quickly, and I stopped following it regularly. I needed something simpler and cooler.

While working on [MCP (Model Context Protocol)](https://github.com/modelcontextprotocol) topics like every other developer these days, it suddenly hit me: what I needed to do was super simple! A basic MCP server would query my PostgreSQL database, and instead of looking at my car's dashboard, I could just talk to it. I rolled up my sleeves, exposed PostgreSQL ports in my compose file, and the rest was just setting up a simple [Python MCP](https://github.com/jlowin/fastmcp) server.

> Everything I'm about to explain can be done with any SQL DB, so you don't necessarily need to own a Tesla. If you have basic Python and SQL knowledge, you can adapt this for your own purposes.
>
> - [Complete source code on GitHub](https://github.com/cobanov/teslamate-mcp) and I'm open to contributions
> - [TeslaMate installation guide](https://docs.teslamate.org/docs/installation/docker)
> - [MCP documentation](https://github.com/modelcontextprotocol)

## Setting Up a Simple Python MCP Server

Actually, I had started writing about MCPs before working on this project and I'm still working on it in parallel, but this idea got me so excited that I took a quick break to jump into this fun project. You'll be able to read my article about MCPs on my blog soon.

For those who haven't delved into MCP at all, I hear it sounds intimidating for some unknown reason. MCPs don't really promise us anything we didn't have before (almost nothing), and if you've done anything involving API interactions while working with language models, I'm sure you'll adapt very quickly. When you get into it, you'll see that it's no different from getting data via API access and feeding it to an LLM – MCPs are just protocols that standardize this process, so you can be sure you won't even have to go through a learning curve when developing.

Just like FastAPI, FastMCP has become a standard if you're developing MCPs in Python – there's a very simple explanation here. I won't go into great detail to keep this post from getting too long, and you'll see it in the project code anyway. We need two things: an appropriate SQL query for your needs, and MCP function code that can call this query.

The TeslaMate application gives us these tables in PostgreSQL:

![entities](https://github.com/cobanov/blog-renderer/blob/main/assets/entities.png?raw=true)

Let's write a simple SQL query to check our car's current status.

First, let's find the latest recorded information for each car. For example, if there are 5 records for a car today, let's take the most recent one. Then let's match this information with car names. This way, we can see the car name, not just the ID. The most interesting part is calculating location. Each car has GPS coordinates, so let's find the nearest address to these coordinates using mathematical distance calculation.

Finally, let's bring all the information together. Car name, battery percentage, remaining range, total mileage, outside temperature, whether A/C is on or off, GPS coordinates, nearest address, and when this information was last updated. Essentially, let's create a summary for each car in a single line like "Car X is currently at this address, battery is at X%, has X km range, outside temperature is X degrees." Let's prepare a useful report for the car tracking system.

```sql
SELECT c.name AS car_name,
       p.battery_level,
       p.rated_battery_range_km,
       p.odometer,
       p.outside_temp,
       p.is_climate_on,
       p.latitude,
       p.longitude,
       a.display_name AS LOCATION,
       a.city,
       a.state,
       p.date AS last_update
FROM positions p
JOIN cars c ON p.car_id = c.id
LEFT JOIN LATERAL
  (SELECT *
   FROM addresses a
   ORDER BY ((p.latitude - a.latitude) ^ 2 + (p.longitude - a.longitude) ^ 2)
   LIMIT 1) a ON TRUE
WHERE p.date =
    (SELECT MAX(date)
     FROM positions p2
     WHERE p2.car_id = p.car_id);
```

The output will look like this:

```json
{
  "car_name": "Cobanov",
  "battery_level": 79,
  "rated_battery_range_km": 309.53,
  "odometer": 6605.009314,
  "outside_temp": 23.5,
  "is_climate_on": false,
  "latitude": 40.761497,
  "longitude": 29.981669,
  "location": "## my address ##",
  "city": "Kocaeli",
  "state": "Kocaeli",
  "last_update": "2025-06-08T19:00:59.925000"
}
```

![current-status](https://github.com/cobanov/blog-renderer/blob/main/assets/current_status_claude.png?raw=true)

No need to explain each one like the previous example. If I want to calculate average efficiency by temperature ranges, I write a query like this:

```sql
SELECT c.name AS car_name,
       CASE
           WHEN d.outside_temp_avg < 0 THEN 'Below 0°C'
           WHEN d.outside_temp_avg BETWEEN 0 AND 10 THEN '0-10°C'
           WHEN d.outside_temp_avg BETWEEN 10 AND 20 THEN '10-20°C'
           WHEN d.outside_temp_avg BETWEEN 20 AND 30 THEN '20-30°C'
           ELSE 'Above 30°C'
       END AS temp_range,
       COUNT(*) AS trip_count,
       AVG(d.distance) AS avg_distance_km,
       AVG((d.start_rated_range_km - d.end_rated_range_km) / NULLIF(d.distance, 0) * 100) AS avg_consumption_pct
FROM drives d
JOIN cars c ON d.car_id = c.id
WHERE d.distance > 0
  AND d.start_rated_range_km > d.end_rated_range_km
  AND d.outside_temp_avg IS NOT NULL
GROUP BY c.name,
         CASE
             WHEN d.outside_temp_avg < 0 THEN 'Below 0°C'
             WHEN d.outside_temp_avg BETWEEN 0 AND 10 THEN '0-10°C'
             WHEN d.outside_temp_avg BETWEEN 10 AND 20 THEN '10-20°C'
             WHEN d.outside_temp_avg BETWEEN 20 AND 30 THEN '20-30°C'
             ELSE 'Above 30°C'
         END
ORDER BY c.name,
         avg_consumption_pct;
```

```json
[
  {
    "car_name": "Cobanov",
    "temp_range": "10-20°C",
    "trip_count": 20,
    "avg_distance_km": 6.44,
    "avg_consumption_pct": 128.75
  },
  {
    "car_name": "Cobanov",
    "temp_range": "20-30°C",
    "trip_count": 78,
    "avg_distance_km": 5.74,
    "avg_consumption_pct": 154.45
  },
  {
    "car_name": "Cobanov",
    "temp_range": "Above 30°C",
    "trip_count": 14,
    "avg_distance_km": 3.66,
    "avg_consumption_pct": 298.66
  }
]
```

And yes, this simple 30-line code snippet in our MCP server – of course, I oversimplified everything in the snippet below, so check out the [project](https://github.com/cobanov/teslamate-mcp), but generally it really is this simple. Write your SQL queries according to your needs, add your functions as needed, and everything becomes quite straightforward.

```python
import psycopg
from psycopg.rows import dict_row
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("teslamate")
con_str = "postgresql://username:password@hostname:port/teslamate"


@mcp.tool()
def get_basic_car_information():
    """
    Get the basic car information for each car.
    """
    with psycopg.connect(con_str, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.name,
                       c.model,
                       c.trim_badging,
                       c.exterior_color,
                       c.marketing_name,
                       cs.enabled,
                       cs.free_supercharging
                FROM cars c
                LEFT JOIN car_settings cs ON c.settings_id = cs.id;
                """
            )
            return cur.fetchall()
```

Finally, you'll add this to the config file in Claude or whatever MCP host you're using, and voilà – now you can talk to your desired database or, in our case, your Tesla!

```json
{
  "mcpServers": {
    "teslamate": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/cobanov-air/Developer/teslamate-mcp",
        "run",
        "main.py"
      ]
    }
  }
}
```

The cool thing is that if you make your prompts more complex, Claude starts making multiple tool calls and comparing the data it gets.

Everything's on GitHub if you want to hack on it:

- **GitHub**: [github.com/cobanov/teslamate-mcp](https://github.com/cobanov/teslamate-mcp)
- **Twitter**: [@cobanov](https://twitter.com/cobanov)
