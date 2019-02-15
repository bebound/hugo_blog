+++
title = "CSRF in Django"
author = ["kk"]
date = 2018-11-07T13:58:00+08:00
tags = ["Python", "Django"]
draft = false
noauthor = true
nocomment = true
nodate = true
nopaging = true
noread = true
+++

CSRF(Cross-site request forgery) is a way to generate fake user request to target website. For example, on a malicious website A, there is a button, click it will send request to www.B.com/logout. When the user click this button, he will logout from website B unconsciously. Logout is not a big problem, but malicious website can generate more dangerous request like money transfer.


## Django CSRF protection {#django-csrf-protection}

Each web framework has different approach to do CSRF protection. In Django, the  validation process is below:

1.  When user login for the first time, Django generate a `=csrf_secret=`, add random salt and encrypt it as A, save A to cookie `=csrftoken=`.
2.  When Django processing tag `={{ csrf_token }}=` or `={% csrf_token %}=`, it read `=csrftoken=` cookie A, reverse it to `=csrf_secret=`, add random salt and encrypt it as B, return corresponding HTML.
3.  When Django receive POST request, it will retrive cookie `=csrftoken=` as A, and tries to get `=csrfmiddlewaretoken=` value B from POST data, if it does not exist, it will get header `=X-CSRFToken=` value as B. Then A and B will be reversed to `=csrf_secret=`. If the values are identical, the validation is passed. Otherwise, a 403 error will raise.


## Django CSRF Usage {#django-csrf-usage}


## Form {#form}

```html
<form>
    {% csrf_token %}
</form>
```


## Single AJAX request {#single-ajax-request}

```js
$.ajax({
    data: {
        csrfmiddlewaretoken: '{{ csrf_token }}'
    },
```


## Multiple AJAX request {#multiple-ajax-request}

```js
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});
```

Ref:

1.  [Cross Site Request Forgery protection](https://docs.djangoproject.com/en/2.1/ref/csrf/)
2.  [csrf.py](https://github.com/django/django/blob/master/django/middleware/csrf.py)
3.  [What's the relationship between csrfmiddlewaretoken and csrftoken?](https://stackoverflow.com/questions/48002861/whats-the-relationship-between-csrfmiddlewaretoken-and-csrftoken)
