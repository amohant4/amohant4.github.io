---
layout: default
title: Tech	
permalink: /tech/
---
<div class="home">  
  <ul class="posts">
    {% for post in site.posts %}
    	{% if post.tag == "tech" %}
	      <li>
	        <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
	        <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
	        <br>
	        {{ post.excerpt }}
	      </li>
    	{% endif %}  
    {% endfor %}
  </ul>
</div>
