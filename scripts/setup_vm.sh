



ansible-playbook -i inventory.yaml playbook.yaml --ask-become-pass

pg_isready -d milabench -h 192.168.2.20 -U milabench_write    
pg_isready -d milabench -h 192.168.2.20 -U milabench_read    


# pip install psycopg2
# import psycopg2
# conn = psycopg2.connect(host="localhost", database="milabench", user="milabench_read", password="1234")




# ssh -R 43022:localhost:22 dave@sulaco.local

# -R [bind_address:]port:host:hostport
# -R [bind_address:]port:local_socket
# -R remote_socket:host:hostport
# -R remote_socket:local_socket
# -R [bind_address:]port
#         Specifies that connections to the given TCP port or Unix
#         socket on the remote (server) host are to be forwarded to
#         the local side.
        
# ssh user@webserver.com -CNL localhost:5432:192.168.1.128:5432

# -C      Requests compression of all data (including stdin,
#         stdout, stderr, and data for forwarded X11, TCP and
#         Unix-domain connections).  The compression algorithm is
#         the same used by gzip(1).  Compression is desirable on
#         modem lines and other slow connections, but will only
#         slow down things on fast networks.  The default value can
#         be set on a host-by-host basis in the configuration
#         files; see the Compression option in ssh_config(5).
# -N      Do not execute a remote command.  This is useful for just
#         forwarding ports.  Refer to the description of
#         SessionType in ssh_config(5) for details.
# -L local_socket:remote_socket
#         Specifies that connections to the given TCP port or Unix
#         socket on the local (client) host are to be forwarded to
#     the given host and port, or Unix socket, on the remote
#     side.  This works by allocating a socket to listen to
#     either a TCP port on the local side, optionally bound to
#     the specified bind_address, or to a Unix socket.
#     Whenever a connection is made to the local port or
#     socket, the connection is forwarded over the secure
#     channel, and a connection is made to either host port
#     hostport, or the Unix socket remote_socket, from the
#     remote machine.



# stop local service to test connectivity
sudo service postgresql stop

pg_isready -d milabench -h 192.168.2.20 -U milabench_write   
# localhost:5432 - no response

# forward localhost to milabenchdb:localhost
ssh milabenchdb -CNL localhost:5432:localhost:5432


pg_isready -d milabench -h localhost -U milabench_write
# localhost:5432 - accepting connection

pg_isready -d milabench -h localhost -U milabench_read    
# localhost:5432 - accepting connection 