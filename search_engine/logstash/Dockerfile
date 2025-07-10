FROM docker.elastic.co/logstash/logstash:8.7.0

# Copy PostgreSQL JDBC driver từ thư mục bin vào lib/jars của Logstash
COPY ./bin/postgresql-42.5.0.jar /usr/share/logstash/logstash-core/lib/jars/

# Copy các file config
COPY ./config/ /usr/share/logstash/config/

CMD ["logstash", "-f", "/usr/share/logstash/config/usersync.conf"]
