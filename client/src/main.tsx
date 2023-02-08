import React, {lazy} from 'react';
import ReactDOM from 'react-dom/client';
import {createBrowserRouter, Navigate, RouterProvider} from "react-router-dom";
import {TransactionProvider} from "./context/TransactionContext";
import App from './App';
import './index.css';
// import "tailwindcss/tailwind.css";
import Home from "./pages/Home/Home";
import ShopIndex from "./pages/Shop/Index";
import GoodsDetail from "./pages/Shop/GoodsDetail";
import SignUp from "./pages/User/SignUp";
import LogIn from "./pages/User/LogIn";
import UserCenter from "./pages/User/UserCenter";
import TransactionRecord from "./pages/User/TransactionRecord";
import Auth from "./layouts/Auth/Auth";
// const ShopIndex = lazy(() => import("./pages/Shop/Index"));
// const host = "import.meta\u200b.env.VITE_SHOP_HOST";
const shopHost = import.meta.env.VITE_SHOP_HOST;
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
        element: <Auth />,
        children: [
            {
                index: true,
                //  path: "/signUp",
                element: <SignUp />
            },
            {
                path: "logIn",
                element: <LogIn />
            }
        ]
    },
    {
        path: "/shop/goods/:id",
        element: <GoodsDetail/>,
        loader: ({params}) => {
            return fetch(`${shopHost}/salesOutlets/goods/recommend/details?goodId=${params.id}`);
        }
    },
    {
        path: "/user/payRecord",
        element: <TransactionRecord/>
    },
    {
        path: "*",
        element: <Navigate to="/"/>
    }
]);

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <TransactionProvider>
        <React.StrictMode>
            <RouterProvider router={routers}/>
        </React.StrictMode>
    </TransactionProvider>
);
