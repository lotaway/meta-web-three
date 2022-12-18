import {FC, useEffect, useState} from 'react';
import {Link} from "react-router-dom";
import './Home.less';
import NavBar from "../../components/NavBar/NavBar";
import PayContract from "../../components/PayContract/PayContract";

export default function Home() {
    const [msg, useMsg] = useState(0);
    useEffect(() => {
        const abortController = new AbortController();
        const counter = setInterval(() => {
            fetch("/message", {
                signal: abortController.signal
            }).then(res => {

            }).catch(err => {
                if (err.name === "AbortError") {
                    console.log("Cancel by abort controller.");
                } else {

                }
            });
        }, 5 * 1000);
        return () => {
            abortController.abort("reRender");
            clearInterval(counter);
        };
    }, []);
    return (
        <div className="min-h-screen home">
            <div className="gradient-bg-welcome">
                <NavBar/>
                <h1 className="text-3xl font-bold underline">Welcome to web3!</h1>
                <Link to="/guide">Guide</Link>
                <p>You have {msg} message.</p>
                <PayContract/>
            </div>
        </div>
    );
}