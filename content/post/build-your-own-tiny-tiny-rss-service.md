+++
title = "Build Your Own Tiny Tiny RSS Service"
author = ["KK"]
date = 2019-06-10T00:25:00+08:00
lastmod = 2019-06-10T21:15:00+08:00
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

After Inoreader change the free plan, which limit the max subscription to 150, I begin to find an alternative. Finally, I found Tiny Tiny RSS. It has a nice website and has the fever API Plugin which was supported by most of the RSS reader APP, so you can read RSS on all of you devices.

This post will tell you how to deploy it on your server.


## Prerequisite {#prerequisite}

You need to install [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/) before using `docker-compose.yml`


## Install docker {#install-docker}

Make a new `ttrss` folder, create `docker-compose.yml` with this content:

```yaml
version: "3"
services:
  database.postgres:
    image: sameersbn/postgresql:latest
    container_name: postgres
    environment:
      - PG_PASSWORD=PWD # please change the password
      - DB_EXTENSION=pg_trgm
    volumes:
      - ~/postgres/data/:/var/lib/postgresql/ # persist postgres data to ~/postgres/data/ on the host
    ports:
      - 5433:5432
    restart: always

  service.rss:
    image: wangqiru/ttrss:latest
    container_name: ttrss
    ports:
      - 181:80
    environment:
      - SELF_URL_PATH=https://RSS.com/ # please change to your own domain
      - DB_HOST=database.postgres
      - DB_PORT=5432
      - DB_NAME=ttrss
      - DB_USER=postgres
      - DB_PASS=PWD # please change the password
      - ENABLE_PLUGINS=auth_internal,fever,api_newsplus # auth_internal is required. Plugins enabled here will be enabled for all users as system plugins
      - SESSION_COOKIE_LIFETIME = 8760
    stdin_open: true
    tty: true
    restart: always
    command: sh -c 'sh /wait-for.sh database.postgres:5432 -- php /configure-db.php && exec s6-svscan /etc/s6/'

  service.mercury: # set Mercury Parser API endpoint to `service.mercury:3000` on TTRSS plugin setting page
    image: wangqiru/mercury-parser-api:latest
    container_name: mercury
    expose:
      - 3000
    ports:
      - 3000:3000
    restart: always
```

Run this command to deploy: `docker-compose up -d`. After it finished, the TTRSS service is running on port `181`, the default account is `admin` with password `password`.

I made minor modification on the yml file, you can find the latest file [here](https://github.com/HenryQW/Awesome-TTRSS).


## Nginx configuration {#nginx-configuration}

If you have a domain and you can use Nginx as reverse proxy to redirect TTRSS to the domain.

```nil
upstream ttrssdev {
    server 127.0.0.1:181;
}

server {
    listen 80;
    server_name  RSS.com;
    return 301 https://RSS.com/$request_uri;
}

server {
    listen 443 ssl;
    gzip on;
    server_name  RSS.com;


    access_log /var/log/nginx/ttrssdev_access.log combined;
    error_log  /var/log/nginx/ttrssdev_error.log;

    location / {
        proxy_redirect off;
        proxy_pass http://ttrssdev;

        proxy_set_header  Host                $http_host;
        proxy_set_header  X-Real-IP           $remote_addr;
        proxy_set_header  X-Forwarded-Ssl     on;
        proxy_set_header  X-Forwarded-For     $proxy_add_x_forwarded_for;
        proxy_set_header  X-Forwarded-Proto   $scheme;
        proxy_set_header  X-Frame-Options     SAMEORIGIN;

        client_max_body_size        100m;
        client_body_buffer_size     128k;

        proxy_buffer_size           4k;
        proxy_buffers               4 32k;
        proxy_busy_buffers_size     64k;
        proxy_temp_file_write_size  64k;
    }
    ssl_certificate /etc/letsencrypt/live/rss.fromkk.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/rss.fromkk.com/privkey.pem; # managed by Certbot

}
```

To enable HTTPS on your website, you can use [certbot](https://certbot.eff.org).


## Fever API and Mercury {#fever-api-and-mercury}

-   Fever
    1.  Check `Enable API: Allows accessing this account through the API` in preference
    2.  Enter a new password for fever in `Plugins - Fever Emulation`
-   Mecury Fulltext Extraction
    1.  Check `mecury-fulltext` plugin in `Preference - Plugins`
    2.  Set Mercury Parser API address to `service.mercury:3000` in `Feeds - Mercury Fulltext settings`


## Update {#update}

Simply run this command to update TTRSS code.

```nil
docker-compose down
docker-compose up -d
```


## APP recommendation {#app-recommendation}

[Reeder 4](https://reederapp.com) works great on my iPad. It's smooth and fast, and is worth every penny.

If you want a free APP, I suggest [Fiery Feeds](http://cocoacake.net/apps/fiery/). I stopped using it after ver 2.2, as it's so lagging. If this issue was fixed, I thought it was the biggest competitor for Reeder 4. For more alternative, read this article: [The Best RSS App for iPhone and iPad](https://thesweetsetup.com/apps/best-rss-app-ipad/).

Ref:

1.  [A ttrss setup guide - Start your own RSS aggregator today](https://henry.wang/2018/04/25/ttrss-docker-plugins-guide.html)
