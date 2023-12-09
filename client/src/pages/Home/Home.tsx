import {createRef, useEffect, useState} from 'react'
import {Link} from "react-router-dom"
import './Home.less'
import NavBar from "../../components/NavBar/NavBar"
import PayContract from "../../components/PayContract/PayContract"
import Card from "../../layouts/Card/Card"
import Input from "../../components/Input/Input"

export default function Home() {
    const [msg, useMsg] = useState<number>(0)
    const [addressTo, setAddressTo] = useState<string>("")
    const inputRef = createRef<HTMLInputElement>()
    useEffect(() => {
        const MESSAGE_DELAY = 5 * 1000
        const abortController = new AbortController()
        const counter = setInterval(() => {
            fetch(`${import.meta.env.VITE_SERVER_HOST}/message`, {
                signal: abortController.signal
            }).then(res => {

            }).catch(err => {
                if (err.name === "AbortError") {
                    console.log("Cancel by abort controller.")
                } else {

                }
            })
        }, MESSAGE_DELAY)
        //  do something with webGPU...
        return () => {
            abortController.abort("reRender")
            clearInterval(counter)
        }
    }, [])
    return (
        <div className="min-h-screen home">
            <div className="gradient-bg-welcome">
                <NavBar/>
                <Card>
                    <Card.Header>Welcome to web3!</Card.Header>
                    <Card.Body>
                        <p>You have {msg} message.</p>
                        <Input ref={inputRef} type="text" value={addressTo}
                               onChange={event => setAddressTo(event.target.value)}/>
                        <PayContract addressTo={addressTo}/>
                    </Card.Body>
                    <Card.Below>
                        <Link to="/guide">Guide</Link>
                    </Card.Below>
                </Card>
            </div>
        </div>
    )
}
