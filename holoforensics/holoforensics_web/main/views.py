from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required

def index(request):
    return render(request, 'index.html')

@login_required
def scenes(request):
    return render(request, 'scenes.html')

@login_required
def analysis_dashboard(request):
    return render(request, 'analysis_dashboard.html')

def simple_tools(request):
    return render(request, 'simple_tools.html')

def user_login(request):
    if request.user.is_authenticated:
        return redirect('scenes')
    
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        
        if username and password:
            user = authenticate(request, username=username, password=password)
            
            if user is not None and user.is_active:
                login(request, user)
                next_url = request.GET.get('next', 'scenes')
                messages.success(request, f'Welcome back, {user.username}!')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please enter both username and password.')
    
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('index')
