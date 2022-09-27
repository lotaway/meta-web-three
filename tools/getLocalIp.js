let os = require('os');

//  获取内网ip地址
function getLocalIp() {
    let map = []
        , iFaces = os.networkInterfaces()
    ;

    for (let dev in iFaces) {
        if (iFaces[dev][1].address.indexOf('192.168.44') !== -1) {
            return iFaces[dev][1].address;
        }
    }

    return map;
}


module.exports = getLocalIp;