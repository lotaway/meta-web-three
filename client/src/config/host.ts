enum Host {
    mainServer = import.meta.env.VITE_SERVER_HOST,
    // shopService: "import.meta\u200b.env.VITE_SHOP_SERVICE_HOST",
    goodsService = import.meta.env.VITE_SHOP_SERVICE_HOST
}
export default Host
