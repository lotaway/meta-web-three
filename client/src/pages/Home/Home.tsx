import {FC, useState} from 'react';
import {Link} from "react-router-dom";
import './Home.less';
import NavBar from "../../components/NavBar/NavBar";
import PayContract from "../../components/PayContract/PayContract";

export default function Home() {
    const [msg, useMsg] = useState(0);
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