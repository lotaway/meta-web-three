const http = require("http");

function pageLoad(url) {
    return new Promise(function (resolve, reject) {
        http.get(url, function (res) {
            let htmlArr = [];

            res.on('data', data => htmlArr.push(data));
            res.on('end', () => resolve(htmlArr.join('')));

        }).on('error', e => reject(e));
    });
}

/**
 * 限制最大并发数加载
 * @param urls {Array} 链接数组
 * @param handler {Function} 处理加载
 * @param limit {Number} 最大并发数
 */
function LimitLoad(urls, handler, limit = 5) {

    if (!this instanceof LimitLoad)
        return new LimitLoad(urls, handler, limit);

    const sequence = [].concat(urls)
        , wrapHandler = function (url) {
            const promise = handler(url).then(item => ({img: item, index: promise}));

            return promise;
        }
    ;
    let promises = new Array(limit)
    ;
    this.limit = limit;

    // 将请求并发到最大数
    promises = sequence.splice(0, this.limit).map(url => wrapHandler(url));
    // 如果已没有更多请求，并发全部
    if (sequence.length <= 0) {
        return Promise.all(promises);
    }

    return sequence
        .reduce((last, url) => last
                .then(() => Promise.race(promises))
                .catch(err => console.log(err))
                .then(res => {
                    let pos = promises.findIndex(item => item === res.index);

                    promises.splice(pos, 1);
                    promises.push(wrapHandler(url));
                })
            , Promise.resolve())
        .then(() => Promise.all(promises));
}

/**
 * 图片加载
 * @param url {string} 图片网络地址
 */
function imgLoad(url) {
    return new Promise((resolve, reject) => {
        let image = new Image();

        image.onload = e => resolve(image);
        image.onerror = reject;
        image.src = url;
    });
}

module.exports = {
    LimitLoad,
    imgLoad,
    pageLoad
};