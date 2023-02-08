import {useEffect} from "react"
import {Link, useLocation} from "react-router-dom"
import "./NavBar.less"
import logo from "../../assets/logo.svg"

export default function Header() {
    const {pathname} = useLocation()
    useEffect(() => {
        console.log(`change location!:${pathname}, need to change view`)
    }, [pathname])
    return (
        <nav className="w-full flex nav-bar">
            <h1 className="md:flex-[0.5] flex-initial ">
                <Link to="/">
                    <object className="logo" type="image/svg+xml" data={logo}/>
                </Link>
            </h1>
            <button type="button">Menu</button>
        </nav>
    )
}
