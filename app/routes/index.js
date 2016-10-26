/*var express = require('express');
var router = express.Router();

/!* GET home page. *!/
router.get('/', function (req, res, next) {
    res.render('index', {title: 'Express'});
});

module.exports = router;*/

//  首页
function start(req, res) {

    res.render('index', {
        pageTitle: '同一个世界，同一个梦想'
    });
    return ("Request handler 'start' was called.");
}
exports.start = start;