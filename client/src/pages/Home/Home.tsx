import {createRef, useEffect, useState} from 'react'
import {Link} from "react-router-dom"
import './Home.less'
import NavBar from "../../components/NavBar/NavBar"
import PayContract from "../../components/PayContract/PayContract"
import Card from "../../layouts/Card/Card"
import Input from "../../components/Input/Input"
import { useTranslation } from 'react-i18next'

export default function Home() {
    const {t} = useTranslation()
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
                    <Card.Header>{t("welcome")}</Card.Header>
                    <Card.Body>
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
