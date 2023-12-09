import React from 'react'
import ReactDOM from 'react-dom/client'
import {createBrowserRouter, Navigate, RouterProvider} from "react-router-dom"
import './index.sass'
// import "tailwindcss/tailwind.css"
import './locale/config'
import {AppStoreProvider} from "./store/container"
import {BlockChainProvider} from "./context/BlockChainContext"
import App from './App'
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
import Charge from "./pages/User/Charge"
import Withdrawal from "./pages/User/Withdrawal"
import Account from "./pages/User/Account"
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
                path: "user",
                element: <UserCenter/>,
                children: [
                    {
                        path: "account",
                        element: <Account/>,
                        children: [
                            {
                                path: "charge",
                                element: <Charge/>,
                            },
                            {
                                path: "withdrawal",
                                element: <Withdrawal/>,
                            },
                        ]
                    },
                ]
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
        element: <BlockChainProvider>
            <TransactionRecord/>
        </BlockChainProvider>
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
