let fs = require('fs')
    , path = require('path')
    , zipper = require('zip-local')
;

let publicPath = "E:/workspace/download"
    , projectInclude = [
        "temp"
    ]
    , fileExclude = [
        ".idea"
    ]
;

// zipper.sync.zip("E:/workspace/download/cid.txt").compress().save(path.join(__dirname, "./1.zip"));

function goZip(paths) {
    if (!paths.length) return Promise.resolve();

    let dirName = paths.shift()
        , pa = path.join(publicPath, '/', dirName);

    return (new Promise(function (resolve, reject) {
        fs.stat(pa, function (err, ele) {
            if (!err) {
                if (ele.isDirectory()) {
                    let patches = pa.split("\\");

                    // console.log(JSON.stringify(patches));
                    if (projectInclude.includes(patches[patches.length - 1])) {
                        console.log(`${pa}符合条件，正在打包...`);
                        zipper.zip(pa, (error, zipExport) => {
                            if (!error) {
                                let savePath = path.join(__dirname, `./${dirName}.zip`);

                                console.log(`打包保存路径：${savePath}`);
                                zipExport.compress().save(savePath, error => {
                                    if (error) {
                                        console.log(`压缩保存出错：${error}`);
                                        reject(error);
                                    }
                                    else {
                                        console.log(`压缩保存完成`);
                                        resolve(paths);
                                    }
                                });
                            }
                            else {
                                console.log(`出现错误：${error}`);
                                reject(error);
                            }
                        });
                    }
                    else {
                        resolve(paths);
                    }
                }
                else {
                    resolve(paths);
                }
            }
            else {
                reject(err);
            }
        });
    })).then(paths => goZip(paths));
}

fs.readdir(publicPath, {encoding: "utf-8"}, (error, ...params) => {
    if (error) return console.log(error);
    goZip(...params).then(result => {
        console.log(`打包完成：${result}`);
    }).catch(error => console.log(`打包出错：${error}`));
});
console.log("正在打包");