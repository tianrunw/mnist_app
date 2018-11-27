# mnist_app

The application can be run either directly from source or through a docker 
image. In the following introduction we will begin with source. `mnist_app` 
uses Python Imaging Library to read image files, so it's a good idea to 
install it first.
```
$ pip install pillow
```

#### 1. Start Cassandra server through docker container
```
$ docker run -p 9042:9042 cassandra
```

#### 2. Start Flask server
`cd` into the working directory `mnist_app/` and run
```
$ python main.py
```

#### 3. Using the application through curl
To get basic information of the running application, run
```
$ curl http://0.0.0.0/
```
To upload an MNIST image, run
```
$ curl -F file=@tests/zero.png http://0.0.0.0/mnist
```
To upload a Fashion MNIST image, run
```
$ curl -F file=@tests/trouser.png http://0.0.0.0/fashion
```
To upload a movie review, run
```
$ curl -F file=@tests/movie_review.txt http://0.0.0.0/imdb
```
To review the requests just submitted, run
```
$ curl http://0.0.0.0/database
```

#### 4. Run the application as a docker container
The docker image `tianrunw/mnist` does not use a database backend, records of 
the requests are instead stored in-memory. To run the image, type
```
$ docker run -p 4000:80 tianrunw/mnist
```
Notice that as the Flask service is running in a container, we need to use 
the specified port 4000 to access it, e.g.,
```
$ curl http://0.0.0.0:4000/
```
