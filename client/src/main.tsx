import React, {lazy} from 'react';
import ReactDOM from 'react-dom/client';
import {createBrowserRouter, Navigate, RouterProvider} from "react-router-dom";
import App from './App';
import './index.css';
import Home from "./pages/Home/Home";
import ShopIndex from "./pages/Shop/Index";
import GoodsDetail from "./pages/Shop/GoodsDetail";
import UserCenter from "./pages/User/UserCenter";
// const ShopIndex = lazy(() => import("./pages/Shop/Index"));
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
                path: "/user/center",
                element: <UserCenter/>,
                children: []
            }
        ]
    },
    {
        path: "/shop/goods/:id",
        element: <GoodsDetail/>,
        loader: ({params}) => {
            return fetch(`http://demo.8248.net/salesOutlets/goods/recommend/details?goodId=${params.id}`);
        }
    },
    {
        path: "*",
        element: <Navigate to="/"/>
    }
]);

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
        <RouterProvider router={routers}/>
    </React.StrictMode>
);