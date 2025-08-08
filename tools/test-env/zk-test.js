const zookeeper = require('node-zookeeper-client')

function main() {
    const client = zookeeper.createClient('192.168.1.194:2181')

    console.log('Start Zookeeper Test')

    client.once('connected', () => {
        console.log('Successfully Connected to Zookeeper!')
        console.log('Exiting...')
        client.close()
    })

    client.connect()
    client.on("error", (err) => {
        console.log("Error connecting to Zookeeper: ", err)
    })

    console.log("Waiting for connection to Zookeeper")
}

try {
    main()
} catch (err) {
    console.log("main Error: ", err)
}