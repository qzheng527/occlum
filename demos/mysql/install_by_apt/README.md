# Run MySQL on Occlum

[`MySQL`](https://www.mysql.com/) is a widely used open-source relational database management system (RDBMS).
This example has to be running in Ubuntu based Occlum docker image.

### Install MySQL 8 by apt
```
./install_mysql.sh
```

### Run server and client

#### Initialize and start the MySQL server
```
./run_mysql_server.sh
```
This command initializes and runs the server (using `mysqld`) in Occlum.
When completed, the server starts to wait connections.

#### Start the MySQL client and send simple queries
```
./run_mysql_client.sh
```
This command starts the client (using `mysql`) in Occlum and test basic query SQLs.

The network protocol between client and server uses uds(unix domain socket) by default.
More configuration can be tuned and applied in `my.cnf`.
