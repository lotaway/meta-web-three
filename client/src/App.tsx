import {Outlet} from "react-router-dom"
import TabBar from "./components/TabBar/TabBar"
import {useEffect} from "react"
import * as wasm from "../../wasm-ff/pkg"

function App() {
    useEffect(() => {
        // wasm.greet("from wasm")
        const urlInfo = {
            proxyPrevFix: "/twitter/api",
            host: "https://api.twitter.com",
            path: "/oauth/request_token",
        }
        const method = "POST"
        const fullQueryStr = wasm.twitter_signature(
            method,
            urlInfo.host + urlInfo.path,
            "OxhqcUXNEaQUtMMreqvRdYl38",
            `${location.protocol + "//" + location.host}/airdrop/bindAccount`,
        )
        console.log(`twitter signature fullQueryStr: ${fullQueryStr}`)
        fetch(`${urlInfo.proxyPrevFix}${urlInfo.path}?${fullQueryStr}`, {
            method,
        })
            .then(res => {
                console.log(`res: ${res}`)
                return res.json()
            })
            .then(json => {
                console.log(`json: ${json}`)
            })
            .catch(err => {
                console.log(`err: ${err}`)
            })
    }, [])
    return (
        <div className="app">
            <Outlet/>
            <TabBar/>
        </div>
    )
}

export default App