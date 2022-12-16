import {useEffect} from "react";
import {Link, useLocation} from "react-router-dom";

export default function Header() {
    const {pathname} = useLocation();
    useEffect(() => {
        console.log(`change location!:${pathname}, need to change view`);
    }, [pathname]);
    return (
        <nav className="w-full flex nav-bar">
            <h1 className="md:flex-[0.5] flex-initial ">
                <img className="logo" src="" alt="logo"/>
            </h1>
            <button type="button">Menu</button>
        </nav>
    );
}