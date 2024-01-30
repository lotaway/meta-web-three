const http = require("http")

async function main() {
    http.request({
        url: "https://api.twitter.com/oauth/request_token",
        method: "POST"
    }, (err, res) => {
        if (err) {
            console.log(`err!!:${err}`)
        }
        console.log(`Success:: ${res}`)
    })
}

await main()