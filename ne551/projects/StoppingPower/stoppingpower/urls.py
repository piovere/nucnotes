from django.conf.urls import include, url
from django.contrib import admin


urlpatterns = [
	url(r'^cmsp/', include('cmsp.urls')),
	url(r'^admin/', admin.site.urls),
]
