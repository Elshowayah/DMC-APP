#what needs to be done connect to dockers so i can connect to postgres and finish the database.


# running streamlit
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate

streamlit run app.py

change link to csv = 

the program will contain a small little application where people can check in and it goes into the database where 
the data is contained where it will check into the database and populate there name if there is a match and sign 
them into the event. 

The database will contain all the members and there details. Also a seperate table for events
so we can keep adding events and the memebers can be populated easy for every event and for new members they 
are going to be asked to fill out the information from before, name, classfication, v number.... 

that being said we
are going to start importing all the data points (students and the information and creating the table and varabiles
from scratch. 

what ifs:
How can we include a contoiusly updated google sheet that the program imports the new memebers?

how can we connect the events slide with the main database to see if it can pull from it
the program has to also include a way to change the classfication of people as well accurately. After
gradutation as well how can we remove V numbers and change to personal emails? pop up the last semester for every
senior for emails maybe?  

Maybe every first mixer a pop up check to see if the inofrmation from last year is still right

Is this application going to apply with every event and how can we redirect it to the event database
to create a new event 



bash to create folders

mkdir org_attendance
cd org_attendance

# main app folders
mkdir src data tests

# python modules
mkdir src/members src/events src/attendance src/utils

# create __init__.py to make them packages
touch src/__init__.py
touch src/members/__init__.py src/events/__init__.py src/attendance/__init__.py src/utils/__init__.py

# main scripts
touch src/members/manage_members.py
touch src/events/manage_events.py
touch src/attendance/manage_attendance.py
touch src/utils/db.py
touch src/utils/config.py

# entry point
touch main.py

# config + docs
touch requirements.txt README.md .gitignore

# data storage (CSV / SQLite file goes here)
touch data/members.csv
touch data/events.csv
touch data/attendance.csv

# functionalties
main.py → entry point to run commands (like import members, sync attendance).

src/members/manage_members.py → functions to add/update/find members.

src/events/manage_events.py → functions to create/list events.

src/attendance/manage_attendance.py → functions to sync 
attendance.

src/utils/db.py → SQLite connection and schema creation.

src/utils/config.py → constants (file paths, sheet IDs, etc.).

data/*.csv → raw data or exports (if you want flat-file backup).
tests/ → unit tests later.

requirements.txt → external dependencies (e.g., pandas, sqlalchemy, gspread).



Streamlit App
 ├── admin.py
 ├── checkin.py
 ├── db.py      <-- new database layer
 │
 └── PostgreSQL Database (local Docker / managed cloud)
       ├── members (table)
       ├── events (table)
       ├── attendance (table)
       └── databrowser (SQL VIEW auto-joins the above)


# this is how we will deploy the application simple easy and free

[ Laptops / Kiosks / Phones ]
            │  https
            ▼
      ┌────────────────────┐
      │  Streamlit App     │  (Streamlit Community Cloud / Render / Railway)
      │  (admin.py,        │
      │   checkin.py, db.py)│
      └─────────┬──────────┘
                │ psycopg2 (TLS)
                ▼
        ┌───────────────────┐
        │  Managed Postgres │  (Neon / Supabase / RDS)
        └───────────────────┘
How it works
You deploy the Streamlit app from your GitHub repo (host keeps it running).
Set DATABASE_URL as a secret in the host.
Everyone visiting https://checkin.yourdomain.com can check in; data lands in Postgres.
Simple, cheap, perfect for a single venue and modest traffic.



# once DMC becomes a very large thing we probably need to migrate over to AWS Cloud and the infrastucure would look like this

[ Kiosks / Laptops / Phones ]
              │  https
              ▼
       ┌───────────────────┐
       │  DNS + CDN + WAF  │  (Cloudflare/Route53)
       └─────────┬─────────┘
                 │  https
                 ▼
        ┌──────────────────┐
        │ Load Balancer    │  (AWS ALB / Render LB)
        └───────┬──────────┘
        ┌───────▼──────────┐   ┌──────────────────┐
        │ App Container A   │   │ App Container B  │   (Dockerized Streamlit/fastAPI)
        │ (admin/checkin)   │   │ (autoscale)      │
        └───────┬──────────┘   └─────────┬────────┘
                │                        │
                ├───────────┬────────────┘
                │           │
          ┌─────▼───┐   ┌───▼──────┐
          │  Redis  │   │  Object  │  (optional)
          │ (cache) │   │ Storage  │  (S3 for exports/images)
          └────┬────┘   └────┬─────┘
               │  TLS        │
               ▼             ▼
        ┌────────────────────────┐
        │ Managed Postgres       │  (RDS / Cloud SQL / Neon/Supabase)
        │ + automated backups    │
        └────────────────────────┘

           ┌──────────────────────────────────────────┐
           │  Observability: logs/metrics/alerts      │
           │  (CloudWatch/Grafana/Prometheus/Sentry)  │
           └──────────────────────────────────────────┘
Why it’s nice
Scales: multiple app instances behind a load balancer; add autoscaling for rushes.
Resilient: DB backups, optional read-replicas; CDN caches static assets globally.
Fast UX: optional Redis caches hot reads (e.g., event list), S3 holds any files.
Safe: WAF shields you; secrets live in the host’s secret manager.




# connect to an DMC cloud s3 bucket











