// 	jQuery class init
+function () {

    function Jquery(rotes, context) {
        return new $.fn.selector(rotes, context);
    }

    var $ = Jquery;	// get this(Jquery itself)
    $.fn = $.prototype;

    $.fn = {
        // selector can parse the rotes and return an array of matches element
        selector: function (rotes, context) {
            var parts,
                i,
                matches = [];
            if (!rotes || (context && rotes.context != context)) return null;

            if (typeof rotes === 'string') {
                parts = rotes.split(' ');
            }

            i = parts.length - 1;

            while (i--) {
                switch (parts[i]) {
                    case ">":

                        break;
                    default:
                        break;
                }
            }

            return matches;
        },
        //  always return this, make link call can be use
        _this: function () {
            return this;
        },
        // expand new method
        extend: function (arg) {
            if (typeof arg == 'object') {
                for (var attr in arg) {
                    if (typeof arg[attr] === 'function')
                        Jquery.prototype[attr] = arg[attr];
                }
            }
        }
        /*use like $.extend({
         newFunction: function () {
         console.log('new jquery element function')
         }
         })*/
    };

    // use  prototype link and return a new Jquery method but infect is Jquery itself
    $.fn.selector.prototype = $.fn;

    // word experend

    // get element position/size (especial outerWidth , how to get a exs number)
}();