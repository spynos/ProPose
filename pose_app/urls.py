from django.conf.urls import url
from . import views  # add
from django.conf import settings  # add
from django.conf.urls.static import static  # add

urlpatterns = [
    url(r'^$', views.first_view, name='first_view'),
    url(r'^uimage/$', views.uimage, name='uimage'),  # add
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)  # add