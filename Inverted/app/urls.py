from django.urls import path
from . import views

app_name = 'app'
urlpatterns = [
    path('', views.index),
    path('result/', views.result),
    # path('novel/<int:pk>', views.book_detail_view.as_view(), name='book-detail'),
]