import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import { api, type Post } from './api'

const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })

function relativeTime(minutes: number) {
  if (minutes < 60) return `${Math.max(1, minutes)}m`
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h`
  return new Date(Date.now() - minutes * 60000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export function Feed({ paused }: { paused: boolean }) {
  const [posts, setPosts] = useState<Post[]>([])
  const [nearEnd, setNearEnd] = useState(false)
  const [error, setError] = useState(false)
  const [failures, setFailures] = useState(0)
  const sentinel = useRef<HTMLDivElement>(null)
  const loading = useRef(false)
  const viewport = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (paused) return
    const observer = new IntersectionObserver(([entry]) => setNearEnd(entry.isIntersecting),
      { root: viewport.current, rootMargin: '800px' })
    observer.observe(sentinel.current!)
    return () => observer.disconnect()
  }, [paused])

  useEffect(() => {
    if (paused || !nearEnd || loading.current) return
    // Let the observer see the current layout before requesting another post.
    const timer = setTimeout(async () => {
      loading.current = true
      try {
        const post = await api.next()
        loading.current = false
        setError(false)
        setPosts(previous => [...previous, post])
      } catch {
        loading.current = false
        setError(true)
        setFailures(count => count + 1)
      }
    }, error ? 2000 : 32)
    return () => clearTimeout(timer)
  }, [nearEnd, paused, posts.length, error, failures])

  return <div className={`feed-viewport${paused ? ' feed-paused' : ''}`} ref={viewport} inert={paused}><main className="feed">
    <nav className="tabs" aria-label="Feed navigation">
      <strong aria-current="page">For you</strong>
      <Link to="/dashboard">Analytics</Link>
    </nav>
    {posts.map(post => <Tweet key={post.id} post={post} paused={paused} root={viewport.current} />)}
    <div ref={sentinel} className="more" role="status">
      {error ? 'Could not load a post. Retrying…' : <span className="spinner" aria-label="Loading posts" />}
    </div>
  </main></div>
}

function Tweet({ post, paused, root }: { post: Post; paused: boolean; root: HTMLDivElement | null }) {
  const node = useRef<HTMLElement>(null)
  const liked = useRef(false)
  const rated = useRef(false)
  const [selected, setSelected] = useState(false)
  const [missingAvatar, setMissingAvatar] = useState(false)

  useLayoutEffect(() => {
    if (paused || rated.current) return
    let active = true
    const observer = new IntersectionObserver(([entry]) => {
      if (active && !entry.isIntersecting && entry.boundingClientRect.bottom < 0 && !rated.current) {
        rated.current = true
        observer.disconnect()
        void api.rate(post.id, liked.current ? 1 : 0).catch(console.warn)
      }
    }, { root })
    observer.observe(node.current!)
    return () => { active = false; observer.disconnect() }
  }, [post.id, paused, root])

  const count = post.stats.likes + Number(selected)
  return <article className="tweet" ref={node}>
    {missingAvatar ? <span className="avatar" /> : <img className="avatar" alt="" src={post.author.avatar}
      onError={() => setMissingAvatar(true)} />}
    <div className="tweet-body">
      <div className="meta"><strong>{post.author.name}</strong> <span>@{post.author.handle} · {relativeTime(post.stats.minutes_ago)}</span></div>
      <div className="post-text">{post.text.split(/([#@][\p{L}\p{N}_]+)/u).map((part, index) =>
        index % 2 ? <span className="mention" key={index}>{part}</span> : part)}</div>
      <button className={`like ${selected ? 'liked' : ''}`} aria-label="Like" aria-pressed={selected} onClick={() => {
        if (rated.current) return
        liked.current = !liked.current
        setSelected(liked.current)
      }}><span aria-hidden="true">{selected ? '♥' : '♡'}</span> {count > 0 ? compact.format(count) : ''}</button>
    </div>
  </article>
}
