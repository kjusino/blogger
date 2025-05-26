import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
const cargoTomlString = `
[package]
name = "rust-crud-api"
version = "0.1.0"
edition = "2021"

[dependencies]
postgres = "0.19.10"
serde = "1.0"
serde_json = "1.0"
serde_derive = "1.0"
`;
const helloWorldString = `
fn main() {
    println!("Hello, world!");
}
`;
const dockerComposeString = `
services:
    rustapp:
        container_name: rustapp
        image: rust-crud-api-v1
        build:
            context: .
            dockerfile: Dockerfile
            args:
                DATABASE_URL: postgres://postgres:postgres@db:5432/postgres
        ports:
            - '8080:8080'
        depends_on:
            - db
    db:
        container_name: db
        image: postgres:12
        environment:
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
            POSTGRES_DB: postgres
        ports:
            - '5432:5432'
        volumes:
            - pgdata:/var/lib/postgresql/data

volumes:
    pgdata: {}
`;
const dockerfileString = `
# Build Stage
FROM rust:1.75 as builder

WORKDIR /app

ARG DATABASE_URL

ENV DATABASE_URL=$DATABASE_URL

COPY . .

RUN cargo build --release

# Production Stage
FROM debian:bookworm-slim

WORKDIR /usr/local/bin

COPY --from=builder /app/target/release/rust-crud-api .

CMD ["./rust-crud-api"]
`;
const rustAppString = `
use postgres::{Client, NoTls};
use postgres::Error as PgError;
use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::env;

#[macro_use]
extern crate serde_derive;

//Model: User struct with id, name, and email
#[derive(Serialize, Deserialize)]
struct User {
    id: Option<i32>,
    name: String,
    email: String,
}

//DATABASE_URL
const DB_URL : &str = env!("DATABASE_URL");

const OK: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";
const NOT_FOUND: &str = "HTTP/1.1 404 Not Found\r\n\r\n";
const INTERNAL_SERVER_ERROR: &str = "HTTP/1.1 500 Internal Server Error\r\n\r\n";

//main function
fn main(){
    if let Err(e) = set_database() {
        println!("Error: {}", e);
        return;
    }

    let listener = TcpListener::bind(format!("0.0.0.0:8080")).unwrap();
    println!("Server running on port 8080");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_client(stream),
            Err(e) => println!("Error: {}", e),
        }
    }
}

//handle_client function
fn handle_client(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    let mut request = String::new();

    match stream.read(&mut buffer) {
        Ok(size) => {
            request.push_str(String::from_utf8_lossy(&buffer[..size]).as_ref());
            let (status_line, content) = match &*request {
                r if r.starts_with("POST /users") => handle_post_request(r),
                r if r.starts_with("GET /users") => handle_get_all_request(r),
                r if r.starts_with("GET /users/") => handle_get_request(r),
                r if r.starts_with("PUT /users/") => handle_put_request(r),
                r if r.starts_with("DELETE /users/") => handle_delete_request(r),
                _ => (NOT_FOUND.to_string(), "404 Not Found".to_string()),
            };
            stream.write_all(format!("{}{}", status_line, content).as_bytes()).unwrap();
        }
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    }
}

//Controllers

//handle_post_request function
fn handle_post_request(request: &str) -> (String, String) {
    match (get_user_from_request(&request), Client::connect(DB_URL, NoTls)) {
        (Ok(user), Ok(mut client)) => {
            let result = client
                .execute("INSERT INTO users (name, email) VALUES ($1, $2)", &[&user.name, &user.email])
                .unwrap();

            (OK.to_string(), "User created".to_string())
        }
        _ => (INTERNAL_SERVER_ERROR.to_string(), "Internal Server Error".to_string()),
    }
}

//handle get request function
fn handle_get_request(request: &str) -> (String, String) {
    match (get_id(&request).parse::<i32>(), Client::connect(DB_URL, NoTls)) {
        (Ok(id), Ok(mut client)) => 
        match client.query_one("SELECT * FROM users WHERE id = $1", &[&id]) {
            Ok(row) => {
                let user = User {
                    id: row.get(0),
                    name: row.get(1),
                    email: row.get(2),
                };
                
                (OK.to_string(), serde_json::to_string(&user).unwrap())
            }
            _ => (NOT_FOUND.to_string(), "User not found".to_string()),
    }
        _ => (INTERNAL_SERVER_ERROR.to_string(), "Internal Server Error".to_string()),
    }
}

//handle_get_all_request function
fn handle_get_all_request(request: &str) -> (String, String) {
    match Client::connect(DB_URL, NoTls) {
        Ok(mut client) => {
            let mut users = Vec::new();
            for row in client.query("SELECT * FROM users", &[]).unwrap() {
                let user = User {
                    id: row.get(0),
                    name: row.get(1),
                    email: row.get(2),
                };
                users.push(user);
            }
            (OK.to_string(), serde_json::to_string(&users).unwrap())
        }
        _ => (INTERNAL_SERVER_ERROR.to_string(), "Internal Server Error".to_string()),
    }
}

//handle_put_request function
fn handle_put_request(request: &str) -> (String, String) {
    match (get_id(&request).parse::<i32>(), get_user_from_request(&request),Client::connect(DB_URL, NoTls)) {
        (Ok(id), Ok(user), Ok(mut client)) => {
            client.execute("UPDATE users SET name = $1, email = $2 WHERE id = $3", &[&user.name, &user.email, &id]).unwrap();

            (OK.to_string(), "User updated".to_string())
        }
        _ => (INTERNAL_SERVER_ERROR.to_string(), "Internal Server Error".to_string()),
    }
}

//handle_delete_request function
fn handle_delete_request(request: &str) -> (String, String) {
    match (get_id(&request).parse::<i32>(), Client::connect(DB_URL, NoTls)) {
        (Ok(id), Ok(mut client)) => {
            let rows_affected = client.execute("DELETE FROM users WHERE id = $1", &[&id]).unwrap();

            if rows_affected == 0 {
                return (NOT_FOUND.to_string(), "User not found".to_string());
            }
            (OK.to_string(), "User deleted".to_string())
        }
            
        _ => (INTERNAL_SERVER_ERROR.to_string(), "Internal Server Error".to_string()),
    }
}

//set_database function
fn set_database() -> Result<(), PgError> {
    let mut client = Client::connect(DB_URL, NoTls)?;
    client.batch_execute("
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR NOT NULL,
            email VARCHAR NOT NULL
        )"
    )?;
    Ok(())
}

//get_id function
fn get_id(request: &str) -> &str {
    request.split('/').nth(2).unwrap_or_default().split_whitespace().next().unwrap_or_default()
}

//deserialize user from request body with the id
fn get_user_from_request(request: &str) -> Result<User, serde_json::Error> {
    let body = request.split("\r\n\r\n").last().unwrap_or_default();
    let user: User = serde_json::from_str(body)?;
    Ok(user)
}`;
const Rust: ArticleProps = {
    route: '/rust',
    title: 'Getting Rusty',
    pics: ['rustLogo.png'],
    caption:
        'Rust, a programming language with guaranteed memory safety, speed, and high concurrency.',
    createdDate: '2025-05-21',
    tags: [Tags.Computation],
    content: [
        <h2>Learning Rust</h2>,
        <p>
            I want to learn Rust. Low-level system engineers around the world
            love the programming language's guaranteed memory safety and borrow
            checker features. I, a math geek, love that those beloved features
            of the language are guaranteed by the mathematics of the TCS
            research field known as{' '}
            <a href="https://shemesh.larc.nasa.gov/fm/fm-what.html">
                formal methods
            </a>
            .
        </p>,
        <p>
            At Roche Inc, I've built software systems that implement formal
            methods to the software engineering lifecycle for Software as a
            Medical Device. I automated compliance to FDA standards using formal
            verification methods embedded into our SDLC, with my system
            authorizing the final go/no-go designations for new software
            releases going out to patients around the world. I've found it to be
            rigorous and challenging work, and the experience made me curious
            about Rust's implementation of formal methods to mathematically
            (read <i>"provably"</i>) guarantee memory safety in production.
            NullPointerException, who??
        </p>,
        <p>
            To learn the Rust basics, I'm going to make a simple CRUD API with
            Rust, a task I have done at work for years. To get started with
            Rust, I followed these steps:
        </p>,
        <ol>
            <li>
                Installed Rust using rustup (
                <a href="https://rustup.rs/">https://rustup.rs/</a>).
            </li>
            <li>
                Read some of the official Rust book (
                <a href="https://doc.rust-lang.org/book/">
                    https://doc.rust-lang.org/book/
                </a>
                ).
            </li>
            <li>
                Read about Rust's ownership model, lifetimes, and borrow checker
                to write safe and efficient code.
            </li>
        </ol>,
        <h2>Rust CRUD API</h2>,
        <p>
            I learn by doing, so the first step I did was find a youtube video
            of someone else creating a Rust CRUD API. I am not going to reinvent
            the wheel, so I found this video:
        </p>,
        <iframe
            src="https://www.youtube.com/embed/vhNoiBOuW94?si=yzzqMyU63-DTJBiQ"
            title="YouTube video player"
            allow="accelerometer; clipboard-write; encrypted-media; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
            style={{ display: 'block', alignContent: 'center' }}
        ></iframe>,
        <p>
            The video is sped up, so I had to put it on 0.75x speed and
            minimized on my screen so I have space to see VSCode and Docker
            Desktop.
        </p>,
        <p>
            Before jumping into Rust,{' '}
            <a href="https://www.docker.com/get-started">
                I'd recommend also getting Docker working locally{' '}
            </a>{' '}
            (for everything, not just this project). If you're wondering{' '}
            <a href="https://www.docker.com/resources/what-container/">
                what is Docker
            </a>
            , it is a lightweight containerization technology used by engineers
            around the world to neatly package software applications, their
            dependencies, configurations, and everything else needed. Docker
            allows you to easily partition your physical machine's resources
            (compute, memory, i/o) so that whatever happens inside the Docker
            container stays inside the container and does not affect your
            machine. Its honestly so easy to use after the initial learning
            curve, and makes life super easy as an engineer.
        </p>,
        <p>
            CRUD API's have 4 main functionalities:
            <ul>
                <li>
                    <b>C</b>reate a new item and save it in the DB{' '}
                </li>
                <li>
                    <b>R</b>ead the items stored in the DB and return{' '}
                </li>
                <li>
                    <b>U</b>pdate an existing item's properties and save
                </li>
                <li>
                    <b>D</b>elete an existing item from the DB
                </li>
            </ul>
        </p>,
        <p>Ok, now that we have everything ready, let's dive into the code!</p>,
        <h2>1. Create Rust App</h2>,
        <p>
            To create a new Rust app, I opened a new terminal and used the
            following command:
        </p>,
        <pre>
            <code>cargo new rust-crud-api</code>
        </pre>,
        <p>
            This command uses Rust's package manager Cargo to create a new
            directory called <code>rust-crud-api</code> with a basic Rust
            project structure. Inside this directory, you'll find a src folder
            containing a <code>main.rs</code> file, which is the entry point for
            your Rust application. In addition, you'll see a
            <code> Cargo.toml </code> file, which is the manifest file for your
            Rust project. It contains metadata about your project, including its
            name, version, dependencies, and other configuration details.
        </p>,
        <SyntaxHighlighter
            language="toml"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={{
                minWidth: 0,
                fontSize: '0.875rem',
                padding: '0 6%',
                margin: 0,
                boxSizing: 'border-box',
                width: '100%',
                maxWidth: '90vw',
                overflowX: 'auto',
            }}
        >
            {cargoTomlString}
        </SyntaxHighlighter>,
        <p>
            The 4 dependencies are postgres for database stuff and the rest are
            for serializing and deserializing data.{' '}
        </p>,
        <h2>2. Run "Hello, World!" Application</h2>,
        <p>
            To run the application, I navigated to the project directory in
            terminal and used the following command to compile and run the Rust
            code:
        </p>,
        <pre>
            <code>cargo run</code>
        </pre>,
        <SyntaxHighlighter
            language="rust"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={{
                minWidth: 0,
                fontSize: '0.875rem',
                padding: '0 6%',
                margin: 0,
                boxSizing: 'border-box',
                width: '100%',
                maxWidth: '90vw',
                overflowX: 'auto',
            }}
        >
            {helloWorldString}
        </SyntaxHighlighter>,
        <p>
            This command compiles the Rust code above and executes the resulting
            binary. If everything is set up correctly, you should see the
            default "Hello, world!" message in the terminal executed from within
            the main.rs file.
        </p>,
        <h2>3. CRUD API Code</h2>,
        <p>
            I watched the full video and ended up running into some difficulties
            with the videos' suggested rust version, the rust docker image set
            up for a Mac with intel chip, and the final image running in the
            container. After debugging the issues, I ended up with the following
            code for the contents of the <code>main.rs</code> file:
        </p>,
        <SyntaxHighlighter
            language="rust"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={{
                minWidth: 0,
                fontSize: '0.875rem',
                padding: '0 6%',
                margin: 0,
                boxSizing: 'border-box',
                width: '100%',
                maxWidth: '90vw',
                overflowX: 'auto',
            }}
        >
            {rustAppString}
        </SyntaxHighlighter>,
        <h2>4. Docker Time</h2>,
        <p>
            The CRUD api lives completely inside the main.rs file, but thats not
            everything that is needed. API's require long term storage solutions
            known as Databases, which is where Docker comes in. By
            containerizing the application and the database in separate
            containers, we can ensure that the entire API runs consistently
            across different environments/machines.
        </p>,
        <p>
            For the application, you want to first create a docker image with
            everything needed for the app. Once the image is created, we will
            instantiate a container from the image and run it on our local
            docker engine. Below are the contents of my <code>Dockerfile</code>,
            which is used to build the image for the Rust CRUD API:
        </p>,
        <SyntaxHighlighter
            language="dockerfile"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={{
                minWidth: 0,
                fontSize: '0.875rem',
                padding: '0 6%',
                margin: 0,
                boxSizing: 'border-box',
                width: '100%',
                maxWidth: '90vw',
                overflowX: 'auto',
            }}
        >
            {dockerfileString}
        </SyntaxHighlighter>,
        <p>
            For the database, we will use the public postgres image directly
            from Dockerhub and configure its container to our CRUD API using the
            following docker-compose.yml file:
        </p>,
        <SyntaxHighlighter
            language="yml"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={{
                fontSize: '0.875rem',
                padding: 0,
                margin: 0,
                boxSizing: 'border-box',
                maxWidth: '90vw',
                overflowX: 'auto',
                width: '100%',
            }}
        >
            {dockerComposeString}
        </SyntaxHighlighter>,
        <h2>5. Run Everything </h2>,
        <p>
            After creating the docker image for the Rust CRUD API, we can now
            start the two containers using the <code> docker-compose up </code>
            command:
        </p>,
        <img
            src={require(`../articles/pics/rustApi.png`)}
            className="rustApi"
            style={{ width: '100%', height: 'auto' }}
            alt={'Rust API running in Docker'}
        />,
        <p>
            The above screenshot shows both containers running, one for the Rust
            CRUD API and one for the Postgres database. The Rust API is running
            on port 8080, and the Postgres database is running on port 5432. Now
            that both are running, we can test our implementation by sending
            HTTP requests to the API endpoints. For this, some people like to
            use tools like Postman or curl. I prefer Postman because of its
            user-friendly interface.
        </p>,
        <h2>6. Testing the API</h2>,
        <p>
            To test the API, I used Postman to send HTTP requests to the API. As
            we saw in the previous sections, the API has several endpoints for
            creating, reading, updating, and deleting resources.
        </p>,
        <p>
            I started by testing the GET endpoint to retrieve all resources,
            which returned an empty array since no resources were created yet.
            Then, I tested the POST endpoint to create a new resource with my
            user information, which returned a successful message.
        </p>,
        <img
            src={require(`../articles/pics/postUserWithPostman.png`)}
            className="rustApi"
            style={{ width: '100%', height: 'auto' }}
            alt={'Rust API running in Docker'}
        />,
        <p>
            I then tested the GET endpoint again, and this time it returned my
            name and email. To update my information, I used the PUT endpoint
            with my user ID and a new email, which also returned a successful
            message. Finally, I tested the DELETE endpoint to remove my user
            from the DB, which also returned 200 OK. The API was now working as
            expected, and I created, read, updated, and deleted resources
            successfully.
        </p>,
        <img
            src={require(`../articles/pics/getUserWithPostman.png`)}
            className="rustApi"
            style={{ width: '100%', height: 'auto' }}
            alt={'Rust API running in Docker'}
        />,
        <h2> Lessons Learned and Conclusion </h2>,
        <p>
            The purpose of this article was to teach myself the basics of the
            Rust programming language by doing something I am very familiar with
            in many other languages. By creating a simple CRUD API, I was able
            to use popular features of Rust, like its borrow checker and
            ownership model. I learned the basics of referencing with
            <code> & </code> and dereferencing with
            <code> * </code>.
        </p>,
        <p>
            I also love that Rust variables are immutable by default, which is a
            great way to prevent runtime bugs in production. I was also very
            impressed with the Ok/Err pattern for making function calls. In
            essence, every function call returns a Result type, which is either
            Ok with the result or Err with an error message. This makes it easy
            to account for happy path and negative path scenarios in your
            execution flow.
        </p>,
        <p>
            A fun fact I learned was that Rust is the safest, least
            resource-intensive language in (kinda) the world. It is much safer
            than error-prone C/C++, and is much, much more resource efficient
            than Python or Javascript. Rust is also a compiled language, which
            means that the final executable has been optimized for performance.
            This makes Rust one of the greenest programming languages, as it can
            run efficiently on low-powered devices and in resource-constrained
            environments.
        </p>,
        <p>
            Overall, I really enjoyed getting into the nitty-gritty details of
            Rust, and can see how the language's features can help engineers
            write safer and more efficient (concurrent) code. I have no plans on
            using the language professionally yet, but I appreciate the learning
            experience as a curious developer.
        </p>,
    ],
};

export default Rust;
