const path = require('path')
    , fs = require('fs')
;

function walk(dir, options) {
    let results = []
        , list = fs.readdirSync(dir)
        , defaultOptions = {
            exclude: [] //  排除目录
            , ext: []   //  过滤后缀名
            , deep: true    //  深度查找
        }
    ;

    Object.keys(defaultOptions).forEach(key => options[key] = (options[key] === undefined ? defaultOptions[key] : options[key]));
    list.forEach(function (file) {
        if (options.exclude.includes(file)) {
            return false;
        }
        file = dir + '/' + file;
        let stat = fs.statSync(file);

        if (stat && stat.isDirectory()) {
            options.deep ? results = results.concat(walk(file)) : "";
        } else {
            if (!options.ext.length || options.ext.includes(path.extname(file).replace(/^\./, ""))) {
                results.push(path.resolve(__dirname, file));
            }
        }
    });

    return results;
}

function dealScr(arr) {
    arr.forEach(filepath => {
        let fileStr = fs.readFileSync(filepath, 'utf-8');
        //       fileStr = fileStr.replace(/[\n]|[\r]/g, "")
//        fileStr = fileStr.replace(/(\<head\>.*?)(\<script.*?\<\/script\>){1,}(.*\<\/head\>)/g, '$1$3')
        fs.writeFileSync(filepath, fileStr);
    });
}

module.exports = walk;