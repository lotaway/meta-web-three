import React, {lazy} from 'react'
import ReactDOM from 'react-dom/client'
import {createBrowserRouter, Navigate, RouterProvider} from "react-router-dom"
import {AppStoreProvider} from "./store/container";
import {ContactProvider} from "./context/ContactContext"
import App from './App'
import './index.css'
// import "tailwindcss/tailwind.css"
import configHost from "./config/host"
import Home from "./pages/Home/Home"
import ShopIndex from "./pages/Shop/Index"
import GoodsDetail from "./pages/Shop/GoodsDetail"
import SignUp from "./pages/User/SignUp"
import LogIn from "./pages/User/LogIn"
import UserCenter from "./pages/User/UserCenter"
import TransactionRecord from "./pages/User/TransactionRecord"
import NewWorld from "./pages/NewWorld/NewWorld"
import Auth from "./layouts/Auth/Auth"
// const ShopIndex = lazy(() => import("./pages/Shop/Index"))
const routers = createBrowserRouter([
    {
        path: "/",
        element: <App/>,
        children: [
            {
                index: true,
                // path: "home",    //  use index no allow path
                element: <Home/>
            },
            {
                path: "shop",
                element: <ShopIndex/>,
                children: [
                    /*{
                        path: "goods/:id",
                        element: <GoodsDetail/>
                    }*/
                ]
            },
            {
                path: "user/center",
                element: <UserCenter/>,
                children: []
            }
        ]
    },
    {
        path: "/auth",
        element: <Auth/>,
        children: [
            {
                index: true,
                //  path: "/signUp",
                element: <SignUp/>
            },
            {
                path: "logIn",
                element: <LogIn/>
            }
        ]
    },
    {
        path: "/shop/goods/:id",
        element: <GoodsDetail/>,
        loader: ({params}) => {
            return fetch(`${configHost.goodsService}/salesOutlets/goods/recommend/details?goodId=${params.id}`);
        }
    },
    {
        path: "/user/payRecord",
        element: <ContactProvider>
            <TransactionRecord/>
        </ContactProvider>
    },
    {
        path: "/NewWorld",
        element: <NewWorld/>
    },
    {
        path: "*",
        element: <Navigate to="/"/>
    }
])

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <AppStoreProvider>
        <React.StrictMode>
            <RouterProvider router={routers}/>
        </React.StrictMode>
    </AppStoreProvider>
)
