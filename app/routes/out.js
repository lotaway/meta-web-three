/**
 * Created by lw on 2016/6/1.
 */
function start(req, res) {
    req.session.destroy();
    res.redirect('/');
}

exports.start = start;