import {Outlet} from "react-router-dom"
import TabBar from "./components/TabBar/TabBar"

function App() {
    return (
        <div className="app">
            <Outlet/>
            <TabBar/>
        </div>
    )
}

export default App